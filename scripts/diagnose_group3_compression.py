# diagnose_group3_compression.py
"""
group3(Unison)의 고출력 구간 예측 압축(compression) 여부를 진단.
zero-shot(group1+2 학습) vs 기존(group3 단독 학습) 두 방식을 비교해서
zero-shot 도입이 고출력 구간 예측력을 악화시켰는지 확인한다.

실행: python diagnose_group3_compression.py
"""
import numpy as np
import pandas as pd

from config import (
    CAPACITY_KWH, BEST_LOSS_CONFIG, BEST_MODEL_CONFIG, LGBM_PARAMS,
    VALID_RATIO_THRESHOLD, INVALID_SAMPLE_WEIGHT,
)
from prepare_data import get_tabular_data, get_target_xy
from train import (
    _prepare_common, _fit_xgb_with_params, _fit_lgbm_with_params,
    _fit_xgb_zeroshot, _fit_lgbm_zeroshot,
)

TARGET = "kpx_group_3"
SOURCE_TARGETS = ["kpx_group_1", "kpx_group_2"]


def compression_report(label, pred, y_true, capacity):
    """고출력 구간(실제 발전량 기준) 예측 압축 여부를 출력."""
    actual_ratio = y_true / capacity
    pred_ratio = pred / capacity

    print(f"\n--- [{label}] 압축 진단 ---")
    print(f"  전체 예측 최댓값: {pred_ratio.max():.1%} (설비용량 대비)")
    print(f"  전체 예측 평균  : {pred_ratio.mean():.1%}")

    for th in [0.70, 0.80, 0.90]:
        mask = actual_ratio >= th
        n = mask.sum()
        if n < 5:
            print(f"  실제 발전량 {th:.0%} 이상: 표본 {n}개 (너무 적어 스킵)")
            continue
        err_rate = np.abs(pred[mask] - y_true[mask]) / capacity
        over_8pct = (err_rate > 0.08).mean()
        print(f"  실제 발전량 {th:.0%} 이상 (n={n}): "
              f"실제 평균={actual_ratio[mask].mean():.1%}, "
              f"예측 평균={pred_ratio[mask].mean():.1%}, "
              f"평균오차율={err_rate.mean():.3f}, "
              f"8%초과 비중={over_8pct:.1%}")


def main():
    capacity = CAPACITY_KWH[TARGET]
    capacity_source = CAPACITY_KWH[SOURCE_TARGETS[0]]

    print("=== 데이터 로드 (2022~2023 학습 -> 2024 검증) ===")
    train_df, X_train, _, _, _ = get_tabular_data(mode="validation", validation_start="2024-01-01")
    X_train_imp, _ = _prepare_common(train_df, X_train, X_train, train_df)

    X_tr, y_tr = get_target_xy(train_df, X_train_imp, TARGET, subset="fit")
    X_v, y_v = get_target_xy(train_df, X_train_imp, TARGET, subset="validation")

    tr_valid = y_tr >= capacity * VALID_RATIO_THRESHOLD
    sw_tr = np.where(tr_valid, 1.0, INVALID_SAMPLE_WEIGHT)

    cfg = BEST_LOSS_CONFIG[TARGET]
    model_cfg = BEST_MODEL_CONFIG.get(TARGET, {})
    xgb_model_params = model_cfg.get("XGB") or {"random_state": 42, "n_jobs": -1}
    lgbm_params = LGBM_PARAMS.copy()
    lgbm_params.update({"importance_type": "gain", **model_cfg.get("LGBM", {})})

    y_true = y_v.values

    # ---------------------------------------------------------------
    # 1) 기존 방식: group3 단독 학습
    # ---------------------------------------------------------------
    print("\n=== [기존] group3 단독 학습 ===")
    best_xgb_base, _ = _fit_xgb_with_params(
        cfg["XGB"]["loss_name"], cfg["XGB"]["params"], X_tr, y_tr, sw_tr, X_v, y_v, capacity, xgb_model_params
    )
    best_lgb_base, _ = _fit_lgbm_with_params(
        cfg["LGBM"]["loss_name"], cfg["LGBM"]["params"], lgbm_params, X_tr, y_tr, sw_tr, X_v, y_v, capacity
    )
    pred_base = np.clip(
        (best_xgb_base.predict(X_v) + best_lgb_base.predict(X_v)) / 2, 0, capacity
    )
    compression_report("group3 단독(기존)", pred_base, y_true, capacity)

    # ---------------------------------------------------------------
    # 2) zero-shot: group1+2 pooled 학습
    # ---------------------------------------------------------------
    print("\n=== [zero-shot] group1+2 pooled 학습 ===")
    X_src_list, y_src_list, sw_src_list = [], [], []
    for src_target in SOURCE_TARGETS:
        X_src, y_src = get_target_xy(train_df, X_train_imp, src_target, subset="fit")
        src_valid = y_src >= CAPACITY_KWH[src_target] * VALID_RATIO_THRESHOLD
        sw_src = np.where(src_valid, 1.0, INVALID_SAMPLE_WEIGHT)
        X_src_list.append(X_src)
        y_src_list.append(y_src)
        sw_src_list.append(sw_src)
    X_source = pd.concat(X_src_list, ignore_index=True)
    y_source = pd.concat(y_src_list, ignore_index=True)
    sw_source = np.concatenate(sw_src_list)

    best_xgb_zs, _ = _fit_xgb_zeroshot(
        TARGET, capacity, capacity_source, X_source, y_source, sw_source,
        X_v, y_v, cfg["XGB"]["loss_name"], cfg["XGB"]["params"], xgb_model_params
    )
    best_lgb_zs, _ = _fit_lgbm_zeroshot(
        TARGET, capacity, capacity_source, X_source, y_source, sw_source,
        X_v, y_v, cfg["LGBM"]["loss_name"], cfg["LGBM"]["params"], lgbm_params
    )
    pred_zs = np.clip(
        (best_xgb_zs.predict(X_v) + best_lgb_zs.predict(X_v)) / 2, 0, capacity
    )
    compression_report("group3 zero-shot", pred_zs, y_true, capacity)

    # ---------------------------------------------------------------
    # 3) 요약 비교
    # ---------------------------------------------------------------
    print("\n=== 요약 비교 ===")
    print(f"{'':20s} {'예측 최댓값(%)':>15s} {'실제70%+ 평균오차':>18s}")
    for label, pred in [("group3 단독(기존)", pred_base), ("group3 zero-shot", pred_zs)]:
        mask70 = (y_true / capacity) >= 0.70
        err70 = np.abs(pred[mask70] - y_true[mask70]) / capacity
        print(f"{label:20s} {pred.max()/capacity:>14.1%} {err70.mean():>17.3f}")


if __name__ == "__main__":
    main()