"""
2024년 전체를 학습에서 제외(2022~2023만 사용)하고 2024를 예측해서,
실제 2024년 라벨과 날짜를 정확히 매칭해 채점하는 백테스트.
evaluate.py의 "2024 Proxy"(순서 매칭, 연도가 다른 데이터 비교)와 달리
진짜 미래 예측 성능을 정확하게 측정한다.

주의: 학습 데이터가 2022~2023(2년)뿐이라 실제 final 모드(2022~2024, 3년) 제출보다
데이터가 적다. 절대 점수를 리더보드와 직접 비교하지 말고,
"이 방법 vs 저 방법" 상대 비교 용도로만 쓸 것.

실행: python backtest_2024.py
"""
import numpy as np
import pandas as pd

from config import (
    TARGET_COLS, CAPACITY_KWH, BEST_LOSS_CONFIG, BEST_MODEL_CONFIG,
    VALID_RATIO_THRESHOLD, WS_FEATURE_COL, MODEL_DIR, RIDGE_PARAMS, LGBM_PARAMS,
    INVALID_SAMPLE_WEIGHT, RAMP_WS_RANGES, RAMP_SAMPLE_WEIGHT, WS_FEATURE_COL,
)
from prepare_data import get_tabular_data, get_target_xy
from train import (
    _prepare_common, _fit_xgb_with_params, _fit_lgbm_with_params, _fit_poly_ridge,
    _true_ficr, load_regime_config, apply_regime_config,
    _fit_xgb_transfer, _fit_lgbm_transfer,
    _fit_xgb_zeroshot, _fit_lgbm_zeroshot,   # 신규
)

TRANSFER_SOURCE_TARGETS = {"kpx_group_3": ["kpx_group_1", "kpx_group_2"]}


def metric_correct(y_true_dict, y_pred_dict):
    group_nmae, group_ficr, per_group = [], [], {}
    for col in TARGET_COLS:
        actual = np.asarray(y_true_dict[col], dtype=float)
        forecast = np.asarray(y_pred_dict[col], dtype=float)
        capacity = CAPACITY_KWH[col]
        valid = actual >= capacity * VALID_RATIO_THRESHOLD
        actual_v, forecast_v = actual[valid], forecast[valid]
        error_rate = np.abs(forecast_v - actual_v) / capacity
        nmae_g = np.mean(error_rate)
        group_nmae.append(nmae_g)
        unit_price = np.select([error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0)
        earned = np.sum(actual_v * unit_price)
        max_settle = np.sum(actual_v * 4.0)
        ficr_g = earned / max_settle if max_settle > 0 else 0.0
        group_ficr.append(ficr_g)
        per_group[col] = {"one_minus_nmae": 1 - nmae_g, "ficr": ficr_g}
    one_minus_nmae = 1 - np.mean(group_nmae)
    ficr = np.mean(group_ficr)
    return 0.5 * one_minus_nmae + 0.5 * ficr, one_minus_nmae, ficr, per_group


def main(loss_config=None, label="default", use_transfer=False, use_zeroshot=False,
         finetune_round_ratio=0.4, finetune_early_stopping_rounds=150): 
    loss_cfg_all = loss_config or BEST_LOSS_CONFIG
    print("=== 2024년 정확 매칭 백테스트 (2022~2023만 학습 -> 2024 예측) ===")
    train_df, X_train, _, _, _ = get_tabular_data(mode="validation", validation_start="2024-01-01")
    X_train_imp, _ = _prepare_common(train_df, X_train, X_train, train_df)

    try:
        regime_all_cfg = load_regime_config(MODEL_DIR / "regime_config_monthly.json")
    except FileNotFoundError:
        regime_all_cfg = {}
        print("[경고] regime_config_monthly.json 없음 -> 단순평균으로 진행")

    y_true_dict, y_pred_dict = {}, {}
    for target in TARGET_COLS:
        capacity = CAPACITY_KWH[target]
        X_tr, y_tr = get_target_xy(train_df, X_train_imp, target, subset="fit")
        X_v, y_v = get_target_xy(train_df, X_train_imp, target, subset="validation")
        if len(X_v) == 0:
            raise ValueError(f"{target}: 2024년 검증 데이터가 없습니다.")

        tr_valid = y_tr >= capacity * VALID_RATIO_THRESHOLD
        sw_tr = np.where(tr_valid, 1.0, INVALID_SAMPLE_WEIGHT)

        ramp_range = RAMP_WS_RANGES.get(target)
        ws_col = WS_FEATURE_COL.get(target)
        if ramp_range is not None and ws_col in X_tr.columns:
            ws_tr = X_tr[ws_col].to_numpy()
            ramp_mask = (ws_tr >= ramp_range[0]) & (ws_tr < ramp_range[1])
            sw_tr = np.where(ramp_mask, sw_tr * RAMP_SAMPLE_WEIGHT, sw_tr)

        model_cfg = BEST_MODEL_CONFIG.get(target, {})
        cfg = loss_cfg_all[target]
        model_preds = {}

        curve_model, _ = _fit_poly_ridge(X_tr, y_tr, sw_tr, X_v, y_v, capacity, RIDGE_PARAMS)
        model_preds["curve"] = curve_model.predict(X_v)

        xgb_model_params = model_cfg.get("XGB") or {}
        lgbm_params = LGBM_PARAMS.copy()
        lgbm_params.update({"importance_type": "gain", **model_cfg.get("LGBM", {})})

        if (use_transfer or use_zeroshot) and target in TRANSFER_SOURCE_TARGETS:
            # --- group1+2를 source로 묶어서 group3(target)로 사용 (전이/zero-shot 공통) ---
            X_src_list, y_src_list, sw_src_list = [], [], []
            for src_target in TRANSFER_SOURCE_TARGETS[target]:
                X_src, y_src = get_target_xy(train_df, X_train_imp, src_target, subset="fit")
                src_valid = y_src >= CAPACITY_KWH[src_target] * VALID_RATIO_THRESHOLD
                sw_src = np.where(src_valid, 1.0, INVALID_SAMPLE_WEIGHT)
                X_src_list.append(X_src)
                y_src_list.append(y_src)
                sw_src_list.append(sw_src)
            X_source = pd.concat(X_src_list, ignore_index=True)
            y_source = pd.concat(y_src_list, ignore_index=True)
            sw_source = np.concatenate(sw_src_list)
            capacity_source = CAPACITY_KWH[TRANSFER_SOURCE_TARGETS[target][0]]

            if use_zeroshot:
                best_xgb, _ = _fit_xgb_zeroshot(
                    target, capacity, capacity_source, X_source, y_source, sw_source,
                    X_v, y_v, cfg["XGB"]["loss_name"], cfg["XGB"]["params"],
                    xgb_model_params or {"random_state": 42, "n_jobs": -1},
                )
                model_preds["xgb"] = best_xgb.predict(X_v)

                best_lgb, _ = _fit_lgbm_zeroshot(
                    target, capacity, capacity_source, X_source, y_source, sw_source,
                    X_v, y_v, cfg["LGBM"]["loss_name"], cfg["LGBM"]["params"], lgbm_params,
                )
                model_preds["lgbm"] = best_lgb.predict(X_v)
            else:
                best_xgb, _ = _fit_xgb_transfer(
                    target, capacity, capacity_source, X_source, y_source, sw_source,
                    X_tr, y_tr, sw_tr, X_v, y_v,
                    cfg["XGB"]["loss_name"], cfg["XGB"]["params"], xgb_model_params or {"random_state": 42, "n_jobs": -1},
                    finetune_round_ratio=finetune_round_ratio,
                    finetune_early_stopping_rounds=finetune_early_stopping_rounds,
                )
                model_preds["xgb"] = best_xgb.predict(X_v)

                best_lgb, _ = _fit_lgbm_transfer(
                    target, capacity, capacity_source, X_source, y_source, sw_source,
                    X_tr, y_tr, sw_tr, X_v, y_v,
                    cfg["LGBM"]["loss_name"], cfg["LGBM"]["params"], lgbm_params,
                    finetune_round_ratio=finetune_round_ratio,
                    finetune_early_stopping_rounds=finetune_early_stopping_rounds,
                )
                model_preds["lgbm"] = best_lgb.predict(X_v)
        else:
            # --- 기존 경로 (group1, group2, 그리고 use_transfer=False인 group3) ---
            best_xgb, _ = _fit_xgb_with_params(
                cfg["XGB"]["loss_name"], cfg["XGB"]["params"], X_tr, y_tr, sw_tr, X_v, y_v, capacity,
                xgb_model_params or None,
            )
            model_preds["xgb"] = best_xgb.predict(X_v)

            best_lgb, _ = _fit_lgbm_with_params(
                cfg["LGBM"]["loss_name"], cfg["LGBM"]["params"], lgbm_params, X_tr, y_tr, sw_tr, X_v, y_v, capacity
            )
            model_preds["lgbm"] = best_lgb.predict(X_v)

        target_regime = regime_all_cfg.get(target)
        if target_regime:
            ws_col = WS_FEATURE_COL[target]
            pred = apply_regime_config(
                model_preds, X_v[ws_col], train_df.loc[X_v.index, "forecast_kst_dtm"],
                target_regime, capacity, target,
            )
        else:
            pred = np.clip(np.mean(list(model_preds.values()), axis=0), 0, capacity)

        y_true_dict[target] = y_v.values
        y_pred_dict[target] = pred
        print(f"  [{target}] 2024 검증 {len(y_v)}행 예측 완료")

    total, nmae, ficr, per_group = metric_correct(y_true_dict, y_pred_dict)
    print(f"[{label}] Total={total:.4f} | 1-NMAE={nmae:.4f} | FICR={ficr:.4f}")
    for g, v in per_group.items():
        print(f"  [{label}][{g}] 1-NMAE={v['one_minus_nmae']:.4f} | FICR={v['ficr']:.4f}")

    ws_v = X_v[WS_FEATURE_COL["kpx_group_3"]].to_numpy() if target == "kpx_group_3" else None
    if ws_v is not None:
        rated_zone = (ws_v >= 11.0) & (ws_v < 13.0)
        if rated_zone.sum() > 10:
            err_zone = np.abs(pred[rated_zone] - y_v.values[rated_zone]) / capacity
            print(f"  [{label}][rated-zone 11~13m/s] n={rated_zone.sum()}, mean_error_rate={err_zone.mean():.4f}")
    
    return {"label": label, "total": total, "one_minus_nmae": nmae, "ficr": ficr, "per_group": per_group}


if __name__ == "__main__":
    main()