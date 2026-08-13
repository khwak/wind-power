# run_transfer_ablation.py
import evaluate

result_baseline = evaluate.main(loss_config=None, label="group3 단독(현재)",
                                 use_transfer=False, use_zeroshot=False)
result_transfer = evaluate.main(loss_config=None, label="group3 전이학습(patience=150)",
                                 use_transfer=True, finetune_round_ratio=0.6,
                                 finetune_early_stopping_rounds=150)
result_zeroshot = evaluate.main(loss_config=None, label="group3 zero-shot(group1+2만)",
                                 use_zeroshot=True)

print("\n=== 최종 비교 ===")
for r in [result_baseline, result_transfer, result_zeroshot]:
    g3 = r["per_group"]["kpx_group_3"]
    print(f"{r['label']:35s} 1-NMAE={g3['one_minus_nmae']:.4f}  FICR={g3['ficr']:.4f}")