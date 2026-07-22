import argparse
import numpy as np
import pandas as pd
import datetime
import torch
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
    models = {
        "RF": RandomForestRegressor(**RF_PARAMS),
        "ET": ExtraTreesRegressor(**ET_PARAMS),
        "XGB": XGBRegressor(**XGB_PARAMS),
        "LGBM": LGBMRegressor(**LGBM_PARAMS)
    }

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    ensemble_preds = pd.DataFrame(index=sample_sub.index)

    for target in TARGET_COLS:
        train_mask = train_df[target].notna()
        y_train = train_df.loc[train_mask, target]
        preds = []

        for name, model in models.items():
            model.fit(X_train_imp.loc[train_mask], y_train)
            preds.append(model.predict(X_test_imp))
            print(f"  [{target}] {name} Trained.")

        ensemble_preds[target] = np.clip(np.mean(preds, axis=0), 0, CAPACITY_KWH[target])

    return ensemble_preds

def train_chronos(train_df, test_df, sample_sub):
    """chronos 공식 파이프라인 + Rolling Forecast (구간 분할 롤링 예측)"""
    print(f"--- [Mode: Chronos] Rolling Inference with {CHRONOS_MODEL_PATH} ---")
    chronos_preds = pd.DataFrame(index=sample_sub.index)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = BaseChronosPipeline.from_pretrained(
        CHRONOS_MODEL_PATH,
        device_map=device,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    total_prediction_length = len(sample_sub)
    chunk_size = 720  # 30일(720시간) 단위로 끊어서 예측 (1024 한계치 방어)
    max_context_len = 4000  # 유지할 컨텍스트 윈도우 크기
    
    for target in TARGET_COLS:
        # 1. 초기 Context 시계열 설정 (최근 4000스텝 실제 데이터)
        current_context = train_df[target].dropna().values[-max_context_len:]
        target_preds = []
        
        print(f"[{target}] 롤링 예측 시작 (Total: {total_prediction_length} hours, Chunk: {chunk_size} hours)")
        
        # 2. 롤링 예측 루프
        for start_idx in range(0, total_prediction_length, chunk_size):
            current_chunk_size = min(chunk_size, total_prediction_length - start_idx)

            step_num = (start_idx // chunk_size) + 1
            total_steps = (total_prediction_length // chunk_size) + (1 if total_prediction_length % chunk_size else 0)

            context_tensor = (
                torch.from_numpy(current_context.astype(np.float32, copy=False))
                .unsqueeze(0)
                .unsqueeze(0)
            )

            quantiles, mean = pipeline.predict_quantiles(
                inputs=context_tensor,
                prediction_length=current_chunk_size,
                quantile_levels=[0.5],
            )

            chunk_preds = mean[0][0].cpu().numpy()

            assert len(chunk_preds) == current_chunk_size, (
                f"[{target}] Chunk prediction mismatch "
                f"(step={step_num}, start={start_idx}, "
                f"expected={current_chunk_size}, got={len(chunk_preds)})"
            )

            if np.isnan(chunk_preds).any():
                raise ValueError(
                    f"[{target}] NaN detected in chunk prediction "
                    f"(step={step_num})"
                )

            target_preds.extend(chunk_preds)
            current_context = np.concatenate([current_context, chunk_preds])[-max_context_len:]

            print(f"  - Step {step_num}/{total_steps} 완료 (누적 {len(target_preds)}시간 예측)")

        # 최종 예측 길이 검증 및 결과 클리핑(0 미만, 최대 용량 초과 방지)
        assert len(target_preds) == total_prediction_length, (
            f"[{target}] Final prediction mismatch "
            f"(expected={total_prediction_length}, "
            f"got={len(target_preds)})"
        )
            
        chronos_preds[target] = np.clip(target_preds, 0, CAPACITY_KWH[target])
        print(f"[{target}] Chronos-2 Rolling Predicted.\n")
        
    return chronos_preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["chronos", "ensemble", "chronos_ensemble"], default="ensemble")
    args = parser.parse_args()
    
    train_df, X_train, test_df, X_test, sample_sub = get_tabular_data()
    final_preds = pd.DataFrame(index=sample_sub.index)
    
    if args.mode == "ensemble":
        final_preds = train_ensemble(train_df, X_train, X_test, sample_sub)
        
    elif args.mode == "chronos":
        final_preds = train_chronos(train_df, test_df, sample_sub)
        
    elif args.mode == "chronos_ensemble":
        rf_preds = train_ensemble(train_df, X_train, X_test, sample_sub)
        chronos_preds = train_chronos(train_df, test_df, sample_sub)
        
        # 단순 평균 블렌딩 (가중치 조절 가능)
        for col in TARGET_COLS:
            final_preds[col] = (rf_preds[col] * 0.7) + (chronos_preds[col] * 0.3)
            
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