import argparse
import numpy as np
import pandas as pd
import datetime
import torch
import os
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from chronos import BaseChronosPipeline
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from config import TARGET_COLS, CAPACITY_KWH, RF_PARAMS, ET_PARAMS, XGB_PARAMS, LGBM_PARAMS, CHRONOS_MODEL_PATH, OUTPUT_DIR
from prepare_data import get_tabular_data

def train_ensemble(train_df, X_train, X_test, sample_sub):
    print("--- [Mode: Ensemble] Training Multi-Model Ensemble (RF, ET, XGB, LGBM) ---")
    
    lgbm_params = LGBM_PARAMS.copy()
    lgbm_params.update({'importance_type': 'gain'})

    models_dict = {
        "RF": lambda: RandomForestRegressor(**RF_PARAMS),
        "ET": lambda: ExtraTreesRegressor(**ET_PARAMS),
        "XGB": lambda: XGBRegressor(**XGB_PARAMS),
        "LGBM": lambda: LGBMRegressor(**lgbm_params)
    }

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    ensemble_preds = pd.DataFrame(index=sample_sub.index)
    
    raw_feature_importances = pd.DataFrame(index=X_train_imp.columns)
    norm_feature_importances = pd.DataFrame(index=X_train_imp.columns)

    for target in TARGET_COLS:
        train_mask = train_df[target].notna()
        y_train = train_df.loc[train_mask, target]
        preds = []

        for name, model_fn in models_dict.items():
            model = model_fn()
            model.fit(X_train_imp.loc[train_mask], y_train)
            preds.append(model.predict(X_test_imp))
            print(f"  [{target}] {name} Trained.")
            
            col_name = f"{target}_{name}"
            imp_vals = model.feature_importances_
            if name == "LGBM" and imp_vals.sum() > 0:
                imp_vals = imp_vals / imp_vals.sum()
            raw_feature_importances[col_name] = imp_vals
            
            val_min = imp_vals.min()
            val_max = imp_vals.max()
            if val_max > val_min:
                norm_vals = (imp_vals - val_min) / (val_max - val_min)
            else:
                norm_vals = np.zeros_like(imp_vals)
            
            norm_feature_importances[col_name] = norm_vals

        ensemble_preds[target] = np.clip(np.mean(preds, axis=0), 0, CAPACITY_KWH[target])

    raw_feature_importances['mean_importance_norm'] = norm_feature_importances.mean(axis=1)
    sorted_fi = raw_feature_importances.sort_values(by='mean_importance_norm', ascending=False)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fi_path = OUTPUT_DIR / f"feature_importance_calibrated_{timestamp_str}.csv"
    sorted_fi.to_csv(fi_path, encoding='utf-8-sig')
    
    print(f"\n📊 보정된 변수 중요도 저장 완료: {fi_path}")
    return ensemble_preds

def convert_to_autogluon_format(df, is_train=True):
    df = df.copy()

    time_col = "kst_dtm" if "kst_dtm" in df.columns else "forecast_kst_dtm"
    df = df.rename(columns={time_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    covariate_cols = [c for c in df.columns if c not in ["timestamp", "forecast_id"] + TARGET_COLS]

    records = []

    for group in TARGET_COLS:
        group_df = df[["timestamp"] + covariate_cols].copy()
        group_df["item_id"] = group

        if is_train:
            group_df["target"] = df[group]

            last_valid = group_df["target"].last_valid_index()

            if last_valid is not None:
                group_df = group_df.iloc[:last_valid + 1]

        records.append(group_df)

    long_df = pd.concat(records, ignore_index=True)

    ts = TimeSeriesDataFrame.from_data_frame(
        long_df,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    return ts, covariate_cols

def train_autogluon_multivariate(train_df, X_train, test_df, X_test, sample_sub):
    print("--- [Mode: AutoGluon] Training Covariate-Aware Chronos ---")

    train_full = train_df[["forecast_kst_dtm"] + TARGET_COLS].join(X_train)
    test_full = test_df[["forecast_kst_dtm"]].join(X_test)

    train_data, covariate_cols = convert_to_autogluon_format(train_full, True)
    known_covariates, _ = convert_to_autogluon_format(test_full, False)

    prediction_length = int(
        known_covariates.num_timesteps_per_item().iloc[0]
    )

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        target="target",
        freq="h",
        eval_metric="MAE",
        known_covariates_names=covariate_cols,
    )

    predictor.fit(
        train_data,
        hyperparameters={
            "Chronos2": [
                {
                    "model_path": CHRONOS_MODEL_PATH,  # "amazon/chronos-2"
                    "covariate_regressor": "GBM",
                    "target_scaler": "mean_abs",
                },
            ]
        },
        time_limit=1800,  
        random_seed=42,
    )

    predictions = predictor.predict(
        train_data,
        known_covariates=known_covariates
    )

    pred_col = "0.5" if "0.5" in predictions.columns else "mean"

    ag_preds = pd.DataFrame(index=sample_sub.index)

    for target in TARGET_COLS:
        ag_preds[target] = np.clip(
            predictions.loc[target][pred_col].to_numpy(),
            0,
            CAPACITY_KWH[target]
        )

    return ag_preds

def main():
    parser = argparse.ArgumentParser()
    # chronos 단독 대신 autogluon 옵션 추가
    parser.add_argument("--mode", type=str, choices=["autogluon", "ensemble", "ag_ensemble"], default="ensemble")
    args = parser.parse_args()
    
    train_df, X_train, test_df, X_test, sample_sub = get_tabular_data()
    print(test_df.shape)
    print(len(TARGET_COLS))
    print(test_df.head())
    for col in TARGET_COLS:
        s = train_df[col]
        print(col)
        print("NaN 개수:", s.isna().sum())
        print("첫 NaN:", s[s.isna()].index.min())
        print("마지막 NaN:", s[s.isna()].index.max())
    final_preds = pd.DataFrame(index=sample_sub.index)
    
    if args.mode == "ensemble":
        final_preds = train_ensemble(train_df, X_train, X_test, sample_sub)
        
    elif args.mode == "autogluon":
        final_preds = train_autogluon_multivariate(train_df, X_train, test_df, X_test, sample_sub)
        
    elif args.mode == "ag_ensemble":
        rf_preds = train_ensemble(train_df, X_train, X_test, sample_sub)
        ag_preds = train_autogluon_multivariate(train_df, X_train, test_df, X_test, sample_sub)
        
        # 단순 평균 블렌딩 (가중치 조절 가능 - 예: ML 0.7, AG 0.3)
        for col in TARGET_COLS:
            final_preds[col] = (rf_preds[col] * 0.7) + (ag_preds[col] * 0.3)
            
    # 결과 저장
    submission = sample_sub[["forecast_id", "forecast_kst_dtm"]].copy()
    for col in TARGET_COLS:
        submission[col] = final_preds[col]
        
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{args.mode}_{timestamp_str}.csv"
    output_path = OUTPUT_DIR / filename
    
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 파이프라인 완료! 저장 위치: {output_path}")

if __name__ == "__main__":
    main()