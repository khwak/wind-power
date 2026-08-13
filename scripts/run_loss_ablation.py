from config import BEST_LOSS_CONFIG, TARGET_COLS
import evaluate

# seasonal_optuna_search_log.csv에서 실제 탐색된 huber_capacity 최적 delta (근사값 0.1 아님)
TUNED_HUBER_CAPACITY_DELTA = {
    ("kpx_group_1", "XGB"):  0.05101472208866336,   # 원래도 huber_capacity, 그대로
    ("kpx_group_1", "LGBM"): 0.053177690865739126,
    ("kpx_group_2", "XGB"):  0.21282635719883228,
    ("kpx_group_2", "LGBM"): 0.09448342300791049,
    ("kpx_group_3", "XGB"):  0.06925598810528834,   # 원래도 huber_capacity, 그대로
    ("kpx_group_3", "LGBM"): 0.09781801964118501,
}

def make_huber_capacity_variant_tuned(tuned_deltas):
    variant = {}
    for target in TARGET_COLS:
        variant[target] = {}
        for model in ["XGB", "LGBM"]:
            delta = tuned_deltas[(target, model)]
            variant[target][model] = {"loss_name": "huber_capacity", "params": {"delta": delta}}
    return variant

if __name__ == "__main__":
    hc_cfg = make_huber_capacity_variant_tuned(TUNED_HUBER_CAPACITY_DELTA)
    assert BEST_LOSS_CONFIG != hc_cfg

    result_mixed = evaluate.main(loss_config=BEST_LOSS_CONFIG, label="현재(혼합)")
    result_uniform = evaluate.main(loss_config=hc_cfg, label="huber_capacity 통일(튜닝된 delta)")

    print("\n=== 최종 비교 ===")
    for r in [result_mixed, result_uniform]:
        print(f"{r['label']:25s} Total={r['total']:.4f}  1-NMAE={r['one_minus_nmae']:.4f}  FICR={r['ficr']:.4f}")
        for g, v in r["per_group"].items():
            print(f"   {g:15s} 1-NMAE={v['one_minus_nmae']:.4f}  FICR={v['ficr']:.4f}")