# diagnose_group3_season_agreement.py
import json
from config import CAPACITY_KWH, MODEL_DIR, REGIME_WS_BIN_WIDTH, REGIME_MIN_FOLD_BIN_SAMPLES, REGIME_MIN_BIN_SAMPLES
from prepare_data import get_tabular_data
from train import _prepare_common
from discover_regimes import run_cluster_cv, _long_error_table, _fold_agreement, _pooled_bin_error

target = "kpx_group_3"
capacity = CAPACITY_KWH[target]

with open(MODEL_DIR / "seasonal_clusters.json", "r", encoding="utf-8") as f:
    clusters = json.load(f)[target]

train_df, X_train, _, _, _ = get_tabular_data(mode="final")
X_train_imp, _ = _prepare_common(train_df, X_train, X_train, train_df)

for season_name in ["season_1", "season_2"]:   # 균등분할로 collapse된 두 시즌만 우선 확인
    months = clusters[season_name]
    cv_wide = run_cluster_cv(train_df, X_train_imp, target, capacity, months)
    long_df = _long_error_table(cv_wide, capacity, REGIME_WS_BIN_WIDTH)
    n_folds = cv_wide["fold"].nunique()

    fold_agree = _fold_agreement(long_df, REGIME_MIN_FOLD_BIN_SAMPLES, n_folds)
    pooled_err = _pooled_bin_error(long_df, REGIME_MIN_BIN_SAMPLES)

    print(f"\n=== [{season_name}] 월={months}, fold수={n_folds} ===")
    print("pooled 오차율(모델별, bin별):")
    print(pooled_err)
    print("\nfold별 합의 현황 (bin -> {top_model, top_count, agree_ratio}):")
    for b, info in sorted(fold_agree.items()):
        print(f"  ws_bin={b}: {info}")