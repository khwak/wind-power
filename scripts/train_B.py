import argparse
import numpy as np
import pandas as pd
import datetime
import optuna
import lightgbm as lgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from config import (
    TARGET_COLS, CAPACITY_KWH, RF_PARAMS, ET_PARAMS, XGB_PARAMS, LGBM_PARAMS, OUTPUT_DIR,
    VALID_RATIO_THRESHOLD, EXCLUDE_INVALID_ROWS, INVALID_SAMPLE_WEIGHT,
    VAL_HOLDOUT_RATIO, EARLY_STOPPING_ROUNDS,
    OPTUNA_N_TRIALS, OPTUNA_SEED, OPTUNA_SEARCH_SPACE, HUBER_HESS_FLOOR,
    BEST_LOSS_CONFIG,
)
from prepare_data import get_tabular_data

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASELINE_COL = {
    "kpx_group_1": "power_curve_pred_lookup_group1_gfs",
    "kpx_group_2": "power_curve_pred_lookup_group2_gfs",
    "kpx_group_3": "power_curve_pred_lookup_group3_gfs",
}


# ============================================
# 대회 공식 FICR 관련 함수 (기존과 동일, 유지)
# ============================================
def _neg_ficr(y_true, y_pred, capacity):
    valid = y_true >= capacity * VALID_RATIO_THRESHOLD
    if valid.sum() == 0:
        return 0.0
    error_rate = np.abs(y_pred[valid] - y_true[valid]) / capacity
    unit_price = np.select([error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0)
    ficr = np.sum(y_true[valid] * unit_price) / np.sum(y_true[valid] * 4.0)
    return -ficr


def _lgb_ficr(y_true, y_pred, capacity):
    valid = y_true >= capacity * VALID_RATIO_THRESHOLD
    if valid.sum() == 0:
        return "ficr", 0.0, True
    error_rate = np.abs(y_pred[valid] - y_true[valid]) / capacity
    unit_price = np.select([error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0)
    ficr = np.sum(y_true[valid] * unit_price) / np.sum(y_true[valid] * 4.0)
    return "ficr", ficr, True


def _true_ficr(y_true, y_pred, capacity):
    valid = y_true >= capacity * VALID_RATIO_THRESHOLD
    if valid.sum() == 0:
        return 0.0
    error_rate = np.abs(y_pred[valid] - y_true[valid]) / capacity
    unit_price = np.select([error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0)
    return np.sum(y_true[valid] * unit_price) / np.sum(y_true[valid] * 4.0)


# ============================================
# Step 3: 손실함수 3종 (기존과 동일, 유지)
# ============================================
def make_huber_capacity_objective(capacity, delta, hess_floor=HUBER_HESS_FLOOR):
    delta_kwh = delta * capacity
    def _obj(y_true, y_pred, sample_weight=None):
        diff = y_pred - y_true
        denom = np.sqrt(1 + (diff / delta_kwh) ** 2)
        grad = diff / denom
        hess = np.maximum(1 / denom ** 3, hess_floor)
        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
        return grad, hess
    return _obj


def make_threshold_weighted_objective(capacity, amplitude, sigma):
    def _obj(y_true, y_pred, sample_weight=None):
        diff = y_pred - y_true
        e = np.abs(diff) / capacity
        w = 1 + amplitude * (
            np.exp(-((e - 0.06) ** 2) / (2 * sigma ** 2)) +
            np.exp(-((e - 0.08) ** 2) / (2 * sigma ** 2))
        )
        grad = diff * w
        hess = np.ones_like(diff) * w
        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
        return grad, hess
    return _obj


def make_smooth_ficr_objective(capacity, anchor_weight, k):
    def _obj(y_true, y_pred, sample_weight=None):
        diff = y_pred - y_true
        e = np.abs(diff) / capacity
        s1 = 1 / (1 + np.exp(-k * (e - 0.06)))
        s2 = 1 / (1 + np.exp(-k * (e - 0.08)))

        dUP_de = -k * s1 * (1 - s1) - k * s2 * (1 - s2)
        d2UP_de2 = (-k ** 2 * s1 * (1 - s1) * (1 - 2 * s1)) + (-k ** 2 * s2 * (1 - s2) * (1 - 2 * s2))

        sign_diff = np.sign(diff)
        coef = (y_true / capacity) / 4.0

        dLoss_de = -coef * dUP_de
        d2Loss_de2 = -coef * d2UP_de2
        grad_ficr = dLoss_de * sign_diff
        hess_ficr_raw = d2Loss_de2

        grad_anchor = diff
        hess_anchor = np.ones_like(diff)

        grad = anchor_weight * grad_anchor + (1 - anchor_weight) * grad_ficr
        hess_raw = anchor_weight * hess_anchor + (1 - anchor_weight) * hess_ficr_raw
        hess = np.maximum(hess_raw, 1e-3)

        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
        return grad, hess
    return _obj


LOSS_BUILDERS = {
    "huber_capacity": make_huber_capacity_objective,
    "threshold_weighted": make_threshold_weighted_objective,
    "smooth_ficr": make_smooth_ficr_objective,
}


def _lgb_objective_wrapper(base_obj_fn):
    def _wrapped(y_true, y_pred):
        return base_obj_fn(y_true, y_pred, sample_weight=None)
    return _wrapped


def _fit_xgb_with_params(loss_name, params, X_tr, y_tr, sw_tr, X_val, y_val, capacity, baseline_tr, baseline_val):
    obj_fn = LOSS_BUILDERS[loss_name](capacity, **params)
    xgb_kwargs = {k: v for k, v in XGB_PARAMS.items()}
    model = XGBRegressor(
        **xgb_kwargs,
        objective=obj_fn,
        eval_metric=lambda y_true, y_pred: _neg_ficr(y_true, y_pred, capacity),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    model.fit(
        X_tr, y_tr, sample_weight=sw_tr,
        base_margin=baseline_tr,
        eval_set=[(X_val, y_val)],
        base_margin_eval_set=[baseline_val],
        verbose=False,
    )
    val_pred = model.predict(X_val, base_margin=baseline_val)   # ← predict할 때도 반드시 넘겨야 함
    val_ficr = _true_ficr(y_val.values, val_pred, capacity)
    return model, val_ficr


def _fit_lgbm_with_params(loss_name, params, lgbm_params, X_tr, y_tr, sw_tr, X_val, y_val, capacity, baseline_tr, baseline_val):
    base_obj_fn = LOSS_BUILDERS[loss_name](capacity, **params)
    obj_fn = _lgb_objective_wrapper(base_obj_fn)
    model = LGBMRegressor(**lgbm_params, objective=obj_fn)
    model.fit(
        X_tr, y_tr, sample_weight=sw_tr,
        init_score=baseline_tr,
        eval_set=[(X_val, y_val)],
        eval_init_score=[baseline_val],
        eval_metric=lambda y_true, y_pred: _lgb_ficr(y_true, y_pred, capacity),
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    val_pred = model.predict(X_val) + baseline_val.to_numpy()   # ← predict()는 baseline을 자동으로 안 더해줌, 직접 더함
    val_ficr = _true_ficr(y_val.values, val_pred, capacity)
    return model, val_ficr


# ============================================
# Optuna 탐색 관련 (기존과 동일, 전부 유지 - 삭제하지 않음)
# ============================================
def _suggest_params(trial, loss_name):
    space = OPTUNA_SEARCH_SPACE[loss_name]
    params = {}
    for name, spec in space.items():
        if isinstance(spec["low"], int) and isinstance(spec["high"], int) and not spec.get("log", False):
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec.get("log", False) and isinstance(spec["low"], int):
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=True)
        else:
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return params


def _run_optuna_search(model_type, target, capacity, lgbm_params, X_tr, y_tr, sw_tr, X_val, y_val, search_log,
                        baseline_tr, baseline_val, warm_start=None):
    """손실함수별로 별도 study를 돌려 예산을 나누지 않음.
    warm_start: {"huber_capacity": {...}, "threshold_weighted": {...}, "smooth_ficr": {...}} 형태로
    각 손실함수별 시작점을 넣을 수 있음 (그룹별로 다르게 넣어야 효과적)."""
    best_overall = {"val_ficr": -np.inf, "loss_name": None, "params": None}

    trial_budget = {"huber_capacity": 20, "threshold_weighted": 40, "smooth_ficr": 40}

    for loss_name, n_trials_this_loss in trial_budget.items():
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
        )

        if warm_start and loss_name in warm_start:
            study.enqueue_trial(warm_start[loss_name])

        def _obj(trial, loss_name=loss_name):
            params = _suggest_params(trial, loss_name)
            if model_type == "XGB":
                _, val_ficr = _fit_xgb_with_params(loss_name, params, X_tr, y_tr, sw_tr, X_val, y_val, capacity, baseline_tr, baseline_val)
            else:
                _, val_ficr = _fit_lgbm_with_params(loss_name, params, lgbm_params, X_tr, y_tr, sw_tr, X_val, y_val, capacity, baseline_tr, baseline_val)
            trial.set_user_attr("params", params)
            return val_ficr

        study.optimize(_obj, n_trials=n_trials_this_loss, show_progress_bar=False)

        for t in study.trials:
            search_log.append({
                "target": target, "model": model_type, "loss_name": loss_name,
                "params": str(t.user_attrs.get("params")), "val_ficr": t.value,
            })

        if study.best_value > best_overall["val_ficr"]:
            best_overall = {
                "val_ficr": study.best_value,
                "loss_name": loss_name,
                "params": study.best_trial.user_attrs["params"],
            }

    print(f"  [{target}][{model_type}] Optuna 최종 선택: {best_overall['loss_name']} "
          f"params={best_overall['params']} val_ficr={best_overall['val_ficr']:.4f}")

    if model_type == "XGB":
        best_model, _ = _fit_xgb_with_params(best_overall["loss_name"], best_overall["params"], X_tr, y_tr, sw_tr, X_val, y_val, capacity, baseline_tr, baseline_val)
    else:
        best_model, _ = _fit_lgbm_with_params(best_overall["loss_name"], best_overall["params"], lgbm_params, X_tr, y_tr, sw_tr, X_val, y_val, capacity, baseline_tr, baseline_val)

    return {"model": best_model, **best_overall}


def _record_importance(raw_df, norm_df, target, name, imp_vals):
    col_name = f"{target}_{name}"
    raw_df[col_name] = imp_vals
    val_min, val_max = imp_vals.min(), imp_vals.max()
    norm_df[col_name] = (imp_vals - val_min) / (val_max - val_min) if val_max > val_min else np.zeros_like(imp_vals)


def _prepare_common(train_df, X_train, X_test, sample_sub):
    """Optuna/하드코딩 두 함수가 공통으로 쓰는 전처리 (imputer, 결측/무효구간 처리)"""
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    return X_train_imp, X_test_imp


def _split_group_data(train_df, X_train_imp, target, capacity):
    train_mask = train_df[target].notna()
    y_full = train_df.loc[train_mask, target]
    X_full = X_train_imp.loc[train_mask]
    baseline_full = X_full[BASELINE_COL[target]]   # X_train에 이미 있는 피처를 baseline으로도 재사용

    valid_mask = y_full >= capacity * VALID_RATIO_THRESHOLD
    print(f"  [{target}] 전체 {len(y_full)}행 중 무효구간(10%미만) {(~valid_mask).sum()}행 "
          f"({'제외' if EXCLUDE_INVALID_ROWS else f'가중치 {INVALID_SAMPLE_WEIGHT}로 축소'})")

    if EXCLUDE_INVALID_ROWS:
        X_train_use = X_full[valid_mask].reset_index(drop=True)
        y_train_use = y_full[valid_mask].reset_index(drop=True)
        baseline_use = baseline_full[valid_mask].reset_index(drop=True)
        sw_use = np.ones(len(y_train_use))
    else:
        X_train_use = X_full.reset_index(drop=True)
        y_train_use = y_full.reset_index(drop=True)
        baseline_use = baseline_full.reset_index(drop=True)
        sw_use = np.where(valid_mask.reset_index(drop=True), 1.0, INVALID_SAMPLE_WEIGHT)

    split_idx = int(len(y_train_use) * (1 - VAL_HOLDOUT_RATIO))
    X_tr, X_val = X_train_use.iloc[:split_idx], X_train_use.iloc[split_idx:]
    y_tr, y_val = y_train_use.iloc[:split_idx], y_train_use.iloc[split_idx:]
    baseline_tr, baseline_val = baseline_use.iloc[:split_idx], baseline_use.iloc[split_idx:]
    sw_tr = sw_use[:split_idx]

    return X_train_use, y_train_use, sw_use, X_tr, y_tr, sw_tr, X_val, y_val, baseline_tr, baseline_val

# ============================================
# 기존: Optuna 탐색 기반 학습 (그대로 유지, 삭제하지 않음)
# ============================================
def train_ensemble(train_df, X_train, X_test, sample_sub):
    print("--- [Mode: Ensemble-Search] Training with Optuna Search ---")

    lgbm_params = LGBM_PARAMS.copy()
    lgbm_params.update({'importance_type': 'gain'})

    X_train_imp, X_test_imp = _prepare_common(train_df, X_train, X_test, sample_sub)

    ensemble_preds = pd.DataFrame(index=sample_sub.index)
    raw_feature_importances = pd.DataFrame(index=X_train_imp.columns)
    norm_feature_importances = pd.DataFrame(index=X_train_imp.columns)
    search_log = []
    final_selection_log = []

    for target in TARGET_COLS:
        capacity = CAPACITY_KWH[target]
        X_train_use, y_train_use, sw_use, X_tr, y_tr, sw_tr, X_val, y_val, baseline_tr, baseline_val = \
            _split_group_data(train_df, X_train_imp, target, capacity)

        preds = []

        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_train_use, y_train_use, sample_weight=sw_use)
        preds.append(rf.predict(X_test_imp))
        print(f"  [{target}] RF Trained.")
        _record_importance(raw_feature_importances, norm_feature_importances, target, "RF", rf.feature_importances_)

        et = ExtraTreesRegressor(**ET_PARAMS)
        et.fit(X_train_use, y_train_use, sample_weight=sw_use)
        preds.append(et.predict(X_test_imp))
        print(f"  [{target}] ET Trained.")
        _record_importance(raw_feature_importances, norm_feature_importances, target, "ET", et.feature_importances_)
        test_baseline = X_test_imp[BASELINE_COL[target]]

        # 그룹별 워밍업 값 (직전 그리드서치/Optuna 최고 기록을 시작점으로)
        warm_start = BEST_LOSS_CONFIG.get(target, {})
        xgb_warm = {warm_start.get("XGB", {}).get("loss_name"): warm_start.get("XGB", {}).get("params")} if "XGB" in warm_start else None
        lgb_warm = {warm_start.get("LGBM", {}).get("loss_name"): warm_start.get("LGBM", {}).get("params")} if "LGBM" in warm_start else None

        print(f"  [{target}] XGB - Optuna 탐색 중...")
        best_xgb = _run_optuna_search("XGB", target, capacity, lgbm_params, X_tr, y_tr, sw_tr, X_val, y_val, search_log,
                                       baseline_tr, baseline_val, warm_start=xgb_warm)
        preds.append(best_xgb["model"].predict(X_test_imp, base_margin=test_baseline))
        final_selection_log.append({"target": target, "model": "XGB", "loss_name": best_xgb["loss_name"],
                                     "params": str(best_xgb["params"]), "val_ficr": best_xgb["val_ficr"]})
        _record_importance(raw_feature_importances, norm_feature_importances, target, "XGB", best_xgb["model"].feature_importances_)

        print(f"  [{target}] LGBM - Optuna 탐색 중...")
        best_lgb = _run_optuna_search("LGBM", target, capacity, lgbm_params, X_tr, y_tr, sw_tr, X_val, y_val, search_log,
                                       baseline_tr, baseline_val, warm_start=lgb_warm)
        preds.append(best_lgb["model"].predict(X_test_imp) + test_baseline.to_numpy())
        final_selection_log.append({"target": target, "model": "LGBM", "loss_name": best_lgb["loss_name"],
                                     "params": str(best_lgb["params"]), "val_ficr": best_lgb["val_ficr"]})
        imp_vals = best_lgb["model"].feature_importances_
        if imp_vals.sum() > 0:
            imp_vals = imp_vals / imp_vals.sum()
        _record_importance(raw_feature_importances, norm_feature_importances, target, "LGBM", imp_vals)

        ensemble_preds[target] = np.clip(np.mean(preds, axis=0), 0, capacity)

    raw_feature_importances['mean_importance_norm'] = norm_feature_importances.mean(axis=1)
    sorted_fi = raw_feature_importances.sort_values(by='mean_importance_norm', ascending=False)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fi_path = OUTPUT_DIR / f"feature_importance_calibrated_{timestamp_str}.csv"
    sorted_fi.to_csv(fi_path, encoding='utf-8-sig')
    print(f"\n📊 보정된 변수 중요도 저장 완료: {fi_path}")

    search_log_df = pd.DataFrame(search_log)
    search_log_path = OUTPUT_DIR / f"optuna_search_full_{timestamp_str}.csv"
    search_log_df.to_csv(search_log_path, index=False, encoding='utf-8-sig')
    print(f"📊 Optuna 전체 시행 로그 저장 완료: {search_log_path} (총 {len(search_log_df)}행)")

    final_log_df = pd.DataFrame(final_selection_log)
    final_log_path = OUTPUT_DIR / f"optuna_final_selection_{timestamp_str}.csv"
    final_log_df.to_csv(final_log_path, index=False, encoding='utf-8-sig')
    print(f"📊 최종 선택 로그 저장 완료: {final_log_path}")
    print(final_log_df.to_string(index=False))

    return ensemble_preds


# ============================================
# 신규: 하드코딩된 최적 파라미터로 바로 학습 (탐색 없음, 빠름)
# ============================================
def train_ensemble_final(train_df, X_train, X_test, sample_sub):
    print("--- [Mode: Ensemble-Final] Training with Hardcoded Best Params (BEST_LOSS_CONFIG) ---")

    lgbm_params = LGBM_PARAMS.copy()
    lgbm_params.update({'importance_type': 'gain'})

    X_train_imp, X_test_imp = _prepare_common(train_df, X_train, X_test, sample_sub)

    ensemble_preds = pd.DataFrame(index=sample_sub.index)
    raw_feature_importances = pd.DataFrame(index=X_train_imp.columns)
    norm_feature_importances = pd.DataFrame(index=X_train_imp.columns)
    final_used_log = []

    for target in TARGET_COLS:
        capacity = CAPACITY_KWH[target]
        X_train_use, y_train_use, sw_use, X_tr, y_tr, sw_tr, X_val, y_val, baseline_tr, baseline_val = _split_group_data(
            train_df, X_train_imp, target, capacity
        )

        preds = []

        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_train_use, y_train_use, sample_weight=sw_use)
        preds.append(rf.predict(X_test_imp))
        print(f"  [{target}] RF Trained.")
        _record_importance(raw_feature_importances, norm_feature_importances, target, "RF", rf.feature_importances_)

        et = ExtraTreesRegressor(**ET_PARAMS)
        et.fit(X_train_use, y_train_use, sample_weight=sw_use)
        preds.append(et.predict(X_test_imp))
        print(f"  [{target}] ET Trained.")
        _record_importance(raw_feature_importances, norm_feature_importances, target, "ET", et.feature_importances_)
        test_baseline = X_test_imp[BASELINE_COL[target]]

        cfg = BEST_LOSS_CONFIG[target]

        xgb_cfg = cfg["XGB"]
        best_xgb, xgb_val_ficr = _fit_xgb_with_params(
            xgb_cfg["loss_name"], xgb_cfg["params"], X_tr, y_tr, sw_tr, X_val, y_val, capacity,
            baseline_tr, baseline_val
        )
        preds.append(best_xgb.predict(X_test_imp, base_margin=test_baseline))
        print(f"  [{target}] XGB Trained. loss={xgb_cfg['loss_name']} params={xgb_cfg['params']} val_ficr={xgb_val_ficr:.4f}")
        final_used_log.append({"target": target, "model": "XGB", "loss_name": xgb_cfg["loss_name"],
                                "params": str(xgb_cfg["params"]), "val_ficr": xgb_val_ficr})
        _record_importance(raw_feature_importances, norm_feature_importances, target, "XGB", best_xgb.feature_importances_)

        lgb_cfg = cfg["LGBM"]
        best_lgb, lgb_val_ficr = _fit_lgbm_with_params(
            lgb_cfg["loss_name"], lgb_cfg["params"], lgbm_params, X_tr, y_tr, sw_tr, X_val, y_val, capacity,
            baseline_tr, baseline_val
        )
        preds.append(best_lgb.predict(X_test_imp) + test_baseline.to_numpy())
        print(f"  [{target}] LGBM Trained. loss={lgb_cfg['loss_name']} params={lgb_cfg['params']} val_ficr={lgb_val_ficr:.4f}")
        final_used_log.append({"target": target, "model": "LGBM", "loss_name": lgb_cfg["loss_name"],
                                "params": str(lgb_cfg["params"]), "val_ficr": lgb_val_ficr})
        imp_vals = best_lgb.feature_importances_
        if imp_vals.sum() > 0:
            imp_vals = imp_vals / imp_vals.sum()
        _record_importance(raw_feature_importances, norm_feature_importances, target, "LGBM", imp_vals)

        ensemble_preds[target] = np.clip(np.mean(preds, axis=0), 0, capacity)

    raw_feature_importances['mean_importance_norm'] = norm_feature_importances.mean(axis=1)
    sorted_fi = raw_feature_importances.sort_values(by='mean_importance_norm', ascending=False)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fi_path = OUTPUT_DIR / f"feature_importance_final_{timestamp_str}.csv"
    sorted_fi.to_csv(fi_path, encoding='utf-8-sig')
    print(f"\n📊 변수 중요도 저장 완료: {fi_path}")

    final_log_df = pd.DataFrame(final_used_log)
    final_log_path = OUTPUT_DIR / f"final_config_used_{timestamp_str}.csv"
    final_log_df.to_csv(final_log_path, index=False, encoding='utf-8-sig')
    print(f"📊 사용된 최종 설정 저장 완료: {final_log_path}")
    print(final_log_df.to_string(index=False))

    return ensemble_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["search", "final"], default="final",
                         help="search: Optuna 재탐색 (느림, 시간 있을 때) / final: 하드코딩된 최적값으로 바로 학습 (빠름, 기본값)")
    args = parser.parse_args()

    train_df, X_train, test_df, X_test, sample_sub = get_tabular_data()

    if args.mode == "search":
        final_preds = train_ensemble(train_df, X_train, X_test, sample_sub)
    else:
        final_preds = train_ensemble_final(train_df, X_train, X_test, sample_sub)

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