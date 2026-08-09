"""
다중 fold 계절 CV로 BEST_LOSS_CONFIG 후보를 재산정하는 오프라인 스크립트.
config.py의 BEST_LOSS_CONFIG를 자동으로 덮어쓰지 않는다 — 결과를 보고 수동으로 반영할 것.

실행: python discover_loss_config.py
"""
import json
import pandas as pd

from config import TARGET_COLS, CAPACITY_KWH, LGBM_PARAMS, BEST_LOSS_CONFIG, SEASONAL_CV_FOLD_STARTS, OUTPUT_DIR
from train import _prepare_seasonal_fold_data, run_seasonal_optuna_search

if __name__ == "__main__":
    lgbm_params = LGBM_PARAMS.copy()
    lgbm_params.update({"importance_type": "gain"})

    search_log = []
    proposed_config = {}

    for target in TARGET_COLS:
        print(f"\n{'='*60}\n{target} 다중fold 손실함수 재탐색 시작\n{'='*60}")
        capacity = CAPACITY_KWH[target]
        fold_data = _prepare_seasonal_fold_data(target, capacity, SEASONAL_CV_FOLD_STARTS[target])

        warm_start = BEST_LOSS_CONFIG.get(target, {})
        xgb_warm = {warm_start.get("XGB", {}).get("loss_name"): warm_start.get("XGB", {}).get("params")} if "XGB" in warm_start else None
        lgb_warm = {warm_start.get("LGBM", {}).get("loss_name"): warm_start.get("LGBM", {}).get("params")} if "LGBM" in warm_start else None

        best_xgb = run_seasonal_optuna_search("XGB", target, capacity, lgbm_params, fold_data, search_log, warm_start=xgb_warm)
        best_lgb = run_seasonal_optuna_search("LGBM", target, capacity, lgbm_params, fold_data, search_log, warm_start=lgb_warm)

        proposed_config[target] = {
            "XGB": {
                "loss_name": best_xgb["loss_name"], 
                "params": best_xgb["params"],
                "model_params": best_xgb.get("model_params") 
            },
            "LGBM": {
                "loss_name": best_lgb["loss_name"], 
                "params": best_lgb["params"],
                "model_params": best_lgb.get("model_params")
            },
        }

        print(f"\n[{target}] 기존 BEST_LOSS_CONFIG vs 다중fold 재탐색 제안:")
        print(f"  기존: {BEST_LOSS_CONFIG[target]}")
        print(f"  제안: {proposed_config[target]}")

    search_log_df = pd.DataFrame(search_log)
    search_log_path = OUTPUT_DIR / "seasonal_optuna_search_log.csv"
    search_log_df.to_csv(search_log_path, index=False, encoding="utf-8-sig")
    print(f"\n📊 전체 trial 로그 저장: {search_log_path}")

    proposed_path = OUTPUT_DIR / "proposed_best_loss_config.json"
    with open(proposed_path, "w", encoding="utf-8") as f:
        json.dump(proposed_config, f, ensure_ascii=False, indent=2)
    print(f"📊 제안 config 저장: {proposed_path}")
    print("\n⚠️  config.py의 BEST_LOSS_CONFIG는 자동 반영되지 않았습니다. "
          "위 결과를 확인하고 직접 옮겨 적용하세요.")