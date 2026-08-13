"""
그룹별 풍속 구간-모델 규칙(regime_config.json)을 계절별 CV로 계산해서 저장하는 오프라인 발견 스크립트.
train.py는 여기서 만든 JSON을 읽기만 하고, 실행 시점에 다시 계산하지 않는다.

실행: python discover_regimes.py
(offline.py는 이 스크립트로 대체됩니다.)
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from config import (
    TARGET_COLS, CAPACITY_KWH, SEASONAL_CV_FOLD_STARTS, VALID_RATIO_THRESHOLD,
    REGIME_CONFIG_PATH, REGIME_WS_BIN_WIDTH, REGIME_MIN_BIN_SAMPLES,
    REGIME_MIN_FOLD_BIN_SAMPLES, REGIME_MIN_FOLD_AGREE_RATIO, REGIME_MIN_WIDTH,
    CV_WINDOW_DAYS,
)
from train import run_seasonal_wind_regime_cv, _true_ficr

MODELS_TO_TRY = ["curve", "lgbm", "xgb"]  # 세 그룹 모두 curve 포함해서 검증


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
    pivot_err = agg.pivot(index="ws_bin", columns="model", values="mean_error_rate")
    pivot_n = agg.pivot(index="ws_bin", columns="model", values="n")
    return pivot_err[pivot_n.min(axis=1) >= min_bin_samples].sort_index()


def _find_confident_singles(fold_agreement, pooled_err, min_agree_ratio, min_width, ws_bin_width):
    """fold 합의율이 높고 + 전체 합산 최우수 모델과도 일치하는 bin을 연속 구간으로 병합."""
    confident_model = {}
    for b in sorted(pooled_err.index):
        fa = fold_agreement.get(b)
        if fa is None:
            continue
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
        if w.sum() == 0:
            return 0.0
        w = w / w.sum()
        blended = np.average(arr, axis=0, weights=w)
        return -_true_ficr(y, blended, capacity)

    result = differential_evolution(neg_ficr, bounds=[(0.0, 1)] * len(names),
                                     seed=seed, maxiter=150, tol=1e-6, polish=True)
    w = np.clip(result.x, 0.05, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(names)) / len(names)
    
    rounded = {n: round(float(wi), 4) for n, wi in zip(names, w)}
    diff = 1.0 - sum(rounded.values())
    if diff != 0:
        max_key = max(rounded, key=rounded.get)
        rounded[max_key] = round(rounded[max_key] + diff, 4)
        
    out = rounded
    for m in model_names:
        out.setdefault(m, 0.0)
    return out

def _fit_blend_weights_lofo(cv_wide, model_names, lo, hi, capacity, seed=42, min_fold_samples=10):
    """Leave-one-fold-out 검증: fold를 하나씩 남겨두고 나머지 fold로만 가중치를 학습해서
    그 fold(못 본 데이터)에 적용했을 때의 FICR을 측정한다.
    같은 데이터로 '경계 발견'과 '가중치 최적화'를 동시에 하는 meta-overfitting 위험 점검용."""
    folds = sorted(cv_wide["fold"].unique())
    ficrs_opt, ficrs_uniform = [], []

    for held_out in folds:
        train_data = cv_wide[(cv_wide["fold"] != held_out) & (cv_wide["ws"] >= lo) & (cv_wide["ws"] < hi)]
        eval_data = cv_wide[(cv_wide["fold"] == held_out) & (cv_wide["ws"] >= lo) & (cv_wide["ws"] < hi)]
        if len(train_data) < 30 or len(eval_data) < min_fold_samples:
            continue

        w = _fit_blend_weights(train_data, model_names, lo, hi, capacity, seed=seed)
        names = [m for m in model_names if f"pred_{m}" in eval_data.columns]
        y_eval = eval_data["y_true"].to_numpy()
        arr_eval = np.array([eval_data[f"pred_{n}"].to_numpy() for n in names])

        w_arr = np.array([w.get(n, 0.0) for n in names])
        w_arr = w_arr / w_arr.sum() if w_arr.sum() > 0 else np.ones(len(names)) / len(names)
        blended = np.average(arr_eval, axis=0, weights=w_arr)
        ficrs_opt.append(_true_ficr(y_eval, blended, capacity))
        ficrs_uniform.append(_true_ficr(y_eval, np.mean(arr_eval, axis=0), capacity))

    return ficrs_opt, ficrs_uniform


def _fit_blend_weights_safe(cv_wide, model_names, lo, hi, capacity, seed=42, label=""):
    """_fit_blend_weights의 안전 버전. LOFO로 검증해서, 최적화 가중치가 못 본 fold에서도
    균등가중치보다 확실히 나을 때만 채택하고, 아니면 균등가중치로 폴백한다."""
    opt_weights = _fit_blend_weights(cv_wide, model_names, lo, hi, capacity, seed=seed)
    ficrs_opt, ficrs_uniform = _fit_blend_weights_lofo(cv_wide, model_names, lo, hi, capacity, seed=seed)

    if not ficrs_opt:
        print(f"    [{label}] LOFO 표본 부족 -> 검증 불가, 최적화 가중치 그대로 사용")
        return opt_weights

    mean_opt, mean_uniform = np.mean(ficrs_opt), np.mean(ficrs_uniform)
    print(f"    [{label}] LOFO 검증: 최적화={mean_opt:.4f} vs 균등={mean_uniform:.4f} ({len(ficrs_opt)}개 fold)")

    if mean_opt >= mean_uniform:
        return opt_weights
    print(f"    [{label}] -> 최적화 가중치가 못 본 fold에서 균등가중치보다 못함, 균등가중치로 폴백")
    names = [m for m in model_names if f"pred_{m}" in cv_wide.columns]
    return {m: (round(1.0 / len(names), 4) if m in names else 0.0) for m in model_names}


def build_regime_config_for_group(target, model_names, fold_starts, window_days,
                                   ws_bin_width, min_bin_samples, min_fold_bin_samples,
                                   min_agree_ratio, min_width):
    capacity = CAPACITY_KWH[target]
    cv_wide = run_seasonal_wind_regime_cv(target, capacity, fold_starts,
                                           window_days=window_days, model_names=model_names)
    long_df = _long_error_table(cv_wide, capacity, ws_bin_width)
    n_folds = cv_wide["fold"].nunique()

    fold_agree = _fold_agreement(long_df, min_fold_bin_samples, n_folds)
    pooled_err = _pooled_bin_error(long_df, min_bin_samples)
    confident = _find_confident_singles(fold_agree, pooled_err, min_agree_ratio, min_width, ws_bin_width)

    regimes = []
    lo_bound = -np.inf
    ws_min = float(pooled_err.index.min())
    ws_max = float(pooled_err.index.max()) + ws_bin_width
    for lo, hi, model in sorted(confident):
        if lo > lo_bound:
            gap_lo = lo_bound if np.isfinite(lo_bound) else ws_min
            regimes.append({
                "lo": lo_bound, "hi": lo, "type": "blend",
                "weights": _fit_blend_weights_safe(cv_wide, model_names, gap_lo, lo, capacity,
                                                    label=f"{target} [{gap_lo:.1f},{lo:.1f})"),
            })
        regimes.append({
            "lo": lo, "hi": hi, "type": "blend",
            "weights": _fit_blend_weights_safe(cv_wide, model_names, lo, hi, capacity,
                                                label=f"{target} [{lo:.1f},{hi:.1f})"),
        })
        lo_bound = hi
    tail_lo = lo_bound if np.isfinite(lo_bound) else ws_min
    regimes.append({
        "lo": lo_bound, "hi": np.inf, "type": "blend",
        "weights": _fit_blend_weights_safe(cv_wide, model_names, tail_lo, ws_max, capacity,
                                            label=f"{target} [{tail_lo:.1f},inf)"),
    })

    return {"models": model_names, "regimes": regimes, "n_folds": n_folds, "fold_starts": list(fold_starts)}


def save_regime_config(all_configs, path):
    def convert(o):
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        if isinstance(o, float) and np.isinf(o):
            return None
        return o
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(all_configs), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    all_configs = {}
    for target in TARGET_COLS:
        print(f"\n{'='*60}\n{target} 구간 규칙 발견 시작\n{'='*60}")
        cfg = build_regime_config_for_group(
            target, MODELS_TO_TRY, SEASONAL_CV_FOLD_STARTS[target], CV_WINDOW_DAYS,
            REGIME_WS_BIN_WIDTH, REGIME_MIN_BIN_SAMPLES, REGIME_MIN_FOLD_BIN_SAMPLES,
            REGIME_MIN_FOLD_AGREE_RATIO, REGIME_MIN_WIDTH,
        )
        all_configs[target] = cfg
        for r in cfg["regimes"]:
            print(f"  [{r['lo']}, {r['hi']}) {r['type']} {r['weights']}")

    save_regime_config(all_configs, REGIME_CONFIG_PATH)
    print(f"\n✅ 저장 완료: {REGIME_CONFIG_PATH}")