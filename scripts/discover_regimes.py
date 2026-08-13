"""
월별 군집화(seasonal_clusters.json)를 반영하여, 각 월별 최적 가중치를 산출하는 스크립트.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.model_selection import KFold

from config import (
    TARGET_COLS, CAPACITY_KWH, REGIME_WS_BIN_WIDTH, REGIME_MIN_BIN_SAMPLES,
    REGIME_MIN_FOLD_BIN_SAMPLES, REGIME_MIN_FOLD_AGREE_RATIO, REGIME_MIN_WIDTH,
    OUTPUT_DIR, MODEL_DIR, BEST_LOSS_CONFIG, XGB_PARAMS, LGBM_PARAMS, RIDGE_PARAMS, WS_FEATURE_COL,
    VALID_RATIO_THRESHOLD, INVALID_SAMPLE_WEIGHT
)
from prepare_data import get_tabular_data, get_target_xy
from train import _prepare_common, _fit_xgb_with_params, _fit_lgbm_with_params, _fit_poly_ridge, _true_ficr

MODELS_TO_TRY = ["curve", "lgbm", "xgb"]

# ----------------- 기존 헬퍼 함수들 유지 -----------------
def _long_error_table(cv_wide, capacity, ws_bin_width):
    rows = []
    model_cols = [c for c in cv_wide.columns if c.startswith("pred_")]
    valid = cv_wide["y_true"] >= capacity * VALID_RATIO_THRESHOLD
    for col in model_cols:
        name = col.replace("pred_", "")
        sub = cv_wide[valid].copy()
        sub["model"] = name
        sub["error_rate"] = (sub[col] - sub["y_true"]).abs() / capacity
        sub["ws_bin"] = np.floor(sub["ws"] / ws_bin_width) * ws_bin_width
        rows.append(sub[["fold", "ws_bin", "model", "error_rate"]])
    return pd.concat(rows, ignore_index=True)

def _fold_agreement(long_df, min_fold_bin_samples, n_folds_total):
    agg = long_df.groupby(["fold", "ws_bin", "model"]).agg(
        n=("error_rate", "size"), mean_error_rate=("error_rate", "mean")
    ).reset_index()
    agg = agg[agg["n"] >= min_fold_bin_samples]
    if agg.empty: return {}
    piv = agg.pivot_table(index=["fold", "ws_bin"], columns="model", values="mean_error_rate")
    piv["best"] = piv.idxmin(axis=1)

    result = {}
    for ws_bin, sub in piv.reset_index().groupby("ws_bin"):
        vc = sub["best"].value_counts()
        result[ws_bin] = {
            "top_model": vc.index[0],
            "top_count": int(vc.iloc[0]),
            "agree_ratio": float(vc.iloc[0]) / n_folds_total,
        }
    return result

def _pooled_bin_error(long_df, min_bin_samples):
    agg = long_df.groupby(["ws_bin", "model"]).agg(
        n=("error_rate", "size"), mean_error_rate=("error_rate", "mean")
    ).reset_index()
    if agg.empty: return pd.DataFrame()
    pivot_err = agg.pivot(index="ws_bin", columns="model", values="mean_error_rate")
    pivot_n = agg.pivot(index="ws_bin", columns="model", values="n")
    return pivot_err[pivot_n.min(axis=1) >= min_bin_samples].sort_index()

def _find_confident_singles(fold_agreement, pooled_err, min_agree_ratio, min_width, ws_bin_width):
    confident_model = {}
    for b in sorted(pooled_err.index):
        fa = fold_agreement.get(b)
        if fa is None: continue
        pooled_best = pooled_err.loc[b].idxmin()
        if fa["agree_ratio"] >= min_agree_ratio and fa["top_model"] == pooled_best:
            confident_model[b] = pooled_best

    regimes = []
    cur_model = cur_start = cur_end = None
    for b in sorted(pooled_err.index):
        m = confident_model.get(b)
        if m == cur_model and m is not None:
            cur_end = b + ws_bin_width
        else:
            if cur_model is not None and (cur_end - cur_start) >= min_width:
                regimes.append((cur_start, cur_end, cur_model))
            cur_model, cur_start, cur_end = m, b, b + ws_bin_width
    if cur_model is not None and (cur_end - cur_start) >= min_width:
        regimes.append((cur_start, cur_end, cur_model))
    return regimes

def _fit_blend_weights(cv_wide, model_names, lo, hi, capacity, seed=42):
    sub = cv_wide[(cv_wide["ws"] >= lo) & (cv_wide["ws"] < hi)]
    names = [m for m in model_names if f"pred_{m}" in sub.columns]
    if len(sub) < 30 or not names:
        return {m: round(1.0 / len(model_names), 4) for m in model_names}

    y = sub["y_true"].to_numpy()
    arr = np.array([sub[f"pred_{n}"].to_numpy() for n in names])

    def neg_ficr(raw_w):
        w = np.clip(raw_w, 0, None)
        if w.sum() == 0: return 0.0
        w = w / w.sum()
        blended = np.average(arr, axis=0, weights=w)
        return -_true_ficr(y, blended, capacity)

    result = differential_evolution(neg_ficr, bounds=[(0.0, 1)] * len(names),
                                     seed=seed, maxiter=150, tol=1e-6, polish=True)
    w = np.clip(result.x, 0.0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(names)) / len(names)
    
    rounded = {n: round(float(wi), 4) for n, wi in zip(names, w)}
    diff = 1.0 - sum(rounded.values())
    if diff != 0:
        max_key = max(rounded, key=rounded.get)
        rounded[max_key] = round(rounded[max_key] + diff, 4)
        
    out = rounded
    for m in model_names: out.setdefault(m, 0.0)
    return out

def _fit_blend_weights_safe(cv_wide, model_names, lo, hi, capacity, seed=42, label=""):
    opt_weights = _fit_blend_weights(cv_wide, model_names, lo, hi, capacity, seed=seed)
    folds = sorted(cv_wide["fold"].unique())
    ficrs_opt, ficrs_uniform = [], []

    for held_out in folds:
        train_data = cv_wide[(cv_wide["fold"] != held_out) & (cv_wide["ws"] >= lo) & (cv_wide["ws"] < hi)]
        eval_data = cv_wide[(cv_wide["fold"] == held_out) & (cv_wide["ws"] >= lo) & (cv_wide["ws"] < hi)]
        if len(train_data) < 30 or len(eval_data) < 10: continue

        w = _fit_blend_weights(train_data, model_names, lo, hi, capacity, seed=seed)
        names = [m for m in model_names if f"pred_{m}" in eval_data.columns]
        y_eval = eval_data["y_true"].to_numpy()
        arr_eval = np.array([eval_data[f"pred_{n}"].to_numpy() for n in names])

        w_arr = np.array([w.get(n, 0.0) for n in names])
        w_arr = w_arr / w_arr.sum() if w_arr.sum() > 0 else np.ones(len(names)) / len(names)
        blended = np.average(arr_eval, axis=0, weights=w_arr)
        ficrs_opt.append(_true_ficr(y_eval, blended, capacity))
        ficrs_uniform.append(_true_ficr(y_eval, np.mean(arr_eval, axis=0), capacity))

    if not ficrs_opt:
        return opt_weights

    mean_opt, mean_uniform = np.mean(ficrs_opt), np.mean(ficrs_uniform)
    if mean_opt >= mean_uniform:
        return opt_weights
    names = [m for m in model_names if f"pred_{m}" in cv_wide.columns]
    return {m: (round(1.0 / len(names), 4) if m in names else 0.0) for m in model_names}

# ----------------- 신규: 시즌(클러스터) 내부 K-Fold CV -----------------
def run_cluster_cv(train_df, X_train_imp, target, capacity, target_months, model_names=("xgb", "lgbm", "curve"), n_splits=4):
    """해당 월(Month)들에 속하는 데이터만 필터링하여 K-Fold 교차검증을 수행합니다."""
    mask = train_df["forecast_kst_dtm"].dt.month.isin(target_months)
    df_season = train_df[mask].reset_index(drop=True)
    X_season = X_train_imp[mask].reset_index(drop=True)
    
    X_all, y_all = get_target_xy(df_season, X_season, target, subset="fit")
    if len(X_all) < 100:
        print(f"  [경고] {target_months}월 데이터가 부족합니다 ({len(X_all)}행)")
        return pd.DataFrame()
        
    # [수정] 시계열 데이터의 미래 참조 방지 및 누수 차단을 위해 TimeSeriesSplit(순차 분할)로 교체
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_rows = []
    
    cfg = BEST_LOSS_CONFIG[target]
    lgbm_params = LGBM_PARAMS.copy()
    lgbm_params.update({'importance_type': 'gain'})
    
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_all)):
        X_tr, y_tr = X_all.iloc[tr_idx].reset_index(drop=True), y_all.iloc[tr_idx].reset_index(drop=True)
        X_v, y_v = X_all.iloc[val_idx].reset_index(drop=True), y_all.iloc[val_idx].reset_index(drop=True)
        
        tr_valid = y_tr >= capacity * VALID_RATIO_THRESHOLD
        sw_tr = np.where(tr_valid, 1.0, INVALID_SAMPLE_WEIGHT)
        
        fold_result = pd.DataFrame({
            "fold": fold,
            "ws": X_v[WS_FEATURE_COL[target]].to_numpy(),
            "y_true": y_v.to_numpy(),
        })
        
        if "xgb" in model_names:
            best_xgb, _ = _fit_xgb_with_params(cfg["XGB"]["loss_name"], cfg["XGB"]["params"], X_tr, y_tr, sw_tr, X_v, y_v, capacity)
            fold_result["pred_xgb"] = best_xgb.predict(X_v)

        if "lgbm" in model_names:
            best_lgb, _ = _fit_lgbm_with_params(cfg["LGBM"]["loss_name"], cfg["LGBM"]["params"], lgbm_params, X_tr, y_tr, sw_tr, X_v, y_v, capacity)
            fold_result["pred_lgbm"] = best_lgb.predict(X_v)

        if "curve" in model_names:
            curve_model, _ = _fit_poly_ridge(X_tr, y_tr, sw_tr, X_v, y_v, capacity, RIDGE_PARAMS)
            fold_result["pred_curve"] = curve_model.predict(X_v)
            
        all_rows.append(fold_result)
        
    return pd.concat(all_rows, ignore_index=True)


def main():
    # 1. 클러스터(시즌) 정보 로드
    cluster_path = MODEL_DIR / "seasonal_clusters.json"
    with open(cluster_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)
        
    # 2. 전체 데이터 로드
    train_df, X_train, _, _, _ = get_tabular_data(mode="final")
    X_train_imp, _ = _prepare_common(train_df, X_train, X_train, train_df)

    final_regime_config = {}

    for target in TARGET_COLS:
        print(f"\n{'='*60}\n{target} 월별(클러스터) 구간 규칙 발견 시작\n{'='*60}")
        capacity = CAPACITY_KWH[target]
        target_clusters = clusters[target]
        
        target_month_config = {}
        
        for season_name, target_months in target_clusters.items():
            print(f"\n▶ 분석 중: {season_name} (해당 월: {target_months})")
            cv_wide = run_cluster_cv(train_df, X_train_imp, target, capacity, target_months)
            
            if cv_wide.empty:
                continue
                
            long_df = _long_error_table(cv_wide, capacity, REGIME_WS_BIN_WIDTH)
            n_folds = cv_wide["fold"].nunique()

            fold_agree = _fold_agreement(long_df, REGIME_MIN_FOLD_BIN_SAMPLES, n_folds)
            pooled_err = _pooled_bin_error(long_df, REGIME_MIN_BIN_SAMPLES)
            if pooled_err.empty:
                continue
                
            min_agree_ratio = (REGIME_MIN_FOLD_AGREE_RATIO.get(target, 0.75)
                    if isinstance(REGIME_MIN_FOLD_AGREE_RATIO, dict)
                    else REGIME_MIN_FOLD_AGREE_RATIO)
            min_width = (REGIME_MIN_WIDTH.get(target, 2.0)
                        if isinstance(REGIME_MIN_WIDTH, dict)
                        else REGIME_MIN_WIDTH)

            confident = _find_confident_singles(fold_agree, pooled_err, min_agree_ratio, min_width, REGIME_WS_BIN_WIDTH)

            regimes = []
            lo_bound = -np.inf
            ws_min = float(pooled_err.index.min())
            ws_max = float(pooled_err.index.max()) + REGIME_WS_BIN_WIDTH
            
            for lo, hi, model in sorted(confident):
                if lo > lo_bound:
                    gap_lo = lo_bound if np.isfinite(lo_bound) else ws_min
                    regimes.append({
                        "lo": lo_bound, "hi": lo, "type": "blend",
                        "weights": _fit_blend_weights_safe(cv_wide, MODELS_TO_TRY, gap_lo, lo, capacity, label=f"[{gap_lo:.1f},{lo:.1f})")
                    })
                regimes.append({
                    "lo": lo, "hi": hi, "type": "blend",
                    "weights": _fit_blend_weights_safe(cv_wide, MODELS_TO_TRY, lo, hi, capacity, label=f"[{lo:.1f},{hi:.1f})")
                })
                lo_bound = hi
                
            tail_lo = lo_bound if np.isfinite(lo_bound) else ws_min
            regimes.append({
                "lo": lo_bound, "hi": np.inf, "type": "blend",
                "weights": _fit_blend_weights_safe(cv_wide, MODELS_TO_TRY, tail_lo, ws_max, capacity, label=f"[{tail_lo:.1f},inf)")
            })
            
            target_month_config[season_name] = {"regimes": regimes}
                
        final_regime_config[target] = target_month_config

    # JSON 저장 시 inf 값을 null로 변환
    def convert(o):
        if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list): return [convert(v) for v in o]
        if isinstance(o, float) and np.isinf(o): return None
        return o
        
    output_path = MODEL_DIR / "regime_config_monthly.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(convert(final_regime_config), f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ 월별 다이나믹 앙상블 설정 저장 완료: {output_path}")

if __name__ == "__main__":
    main()