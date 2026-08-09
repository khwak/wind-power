"""
격자별 ws117 vs 실제 발전량 상관관계를, (1) 전체구간/유효구간 (2) 계절별로 나눠서 확인하는 진단 스크립트.
cutoff 이전 데이터만 쓴다 (팀원 분석과 동일한 제약).

실행: python diagnose_grid_correlation.py
"""
import pandas as pd

from config import RAW_DIR, CAPACITY_KWH, VALID_RATIO_THRESHOLD
from prepare_data import (
    GROUP_META, _parse_times, _assert_one_issue_per_target, _resolve_cutoff,
    impute_ldaps_within_issue_cycle, add_ldaps_grid_features,
    PREDICTION_REFERENCE_OFFSET_HOURS,
)

SEASON_MAP = {
    12: "겨울", 1: "겨울", 2: "겨울",
    3: "봄", 4: "봄", 5: "봄",
    6: "여름", 7: "여름", 8: "여름",
    9: "가을", 10: "가을", 11: "가을",
}


def load_cutoff_eligible_grid_rows(mode="final", validation_start=None):
    """cutoff 이전 (grid_id, forecast_kst_dtm, ws117_power_ldaps) 행을 반환한다."""
    train_labels_all = pd.read_csv(RAW_DIR / "train" / "train_labels.csv", encoding="utf-8-sig")
    ldaps_train = pd.read_csv(RAW_DIR / "train" / "ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv(RAW_DIR / "train" / "gfs_train.csv", encoding="utf-8-sig")
    ldaps_test = pd.read_csv(RAW_DIR / "test" / "ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv(RAW_DIR / "test" / "gfs_test.csv", encoding="utf-8-sig")

    _parse_times(train_labels_all, ldaps_train, gfs_train, ldaps_test, gfs_test)
    for frame, name in ((ldaps_train, "LDAPS TRAIN"), (ldaps_test, "LDAPS TEST"),
                         (gfs_train, "GFS TRAIN"), (gfs_test, "GFS TEST")):
        _assert_one_issue_per_target(frame, name)

    cutoff, _ = _resolve_cutoff(
        mode=mode, validation_start=validation_start,
        ldaps_train=ldaps_train, gfs_train=gfs_train,
        ldaps_test=ldaps_test, gfs_test=gfs_test,
        prediction_reference_offset_hours=PREDICTION_REFERENCE_OFFSET_HOURS,
    )
    print(f"cutoff = {cutoff}")

    ldaps_train_grid = add_ldaps_grid_features(impute_ldaps_within_issue_cycle(ldaps_train))
    fit_rows = ldaps_train_grid[ldaps_train_grid["forecast_kst_dtm"] < cutoff]
    return fit_rows, train_labels_all


def _corr_table(df, target_col, group, scope):
    corr = (
        df.groupby("grid_id")
        .apply(lambda g: g["ws117_power_ldaps"].corr(g[target_col]), include_groups=False)
        .sort_values(ascending=False)
        .reset_index()
    )
    corr.columns = ["grid_id", "correlation"]
    corr["group"] = group
    corr["scope"] = scope
    corr["n"] = len(df)
    corr["rank"] = range(1, len(corr) + 1)
    return corr


def run(mode="final", validation_start=None, min_season_samples=100):
    fit_rows, train_labels_all = load_cutoff_eligible_grid_rows(mode, validation_start)

    all_rows = []
    for group in GROUP_META:
        target_col = f"kpx_group_{group}"
        capacity = CAPACITY_KWH[target_col]
        labels = train_labels_all[["kst_dtm", target_col]].rename(columns={"kst_dtm": "forecast_kst_dtm"})
        merged = fit_rows.merge(labels, on="forecast_kst_dtm", how="inner").dropna(
            subset=["ws117_power_ldaps", target_col]
        )
        merged["season"] = merged["forecast_kst_dtm"].dt.month.map(SEASON_MAP)
        merged["valid"] = merged[target_col] >= capacity * VALID_RATIO_THRESHOLD

        # (1) 전체구간 vs 유효구간
        all_rows.append(_corr_table(merged, target_col, group, "전체구간"))
        all_rows.append(_corr_table(merged[merged["valid"]], target_col, group, "유효구간"))

        # (2) 계절별 (유효구간 기준)
        for season, sub in merged[merged["valid"]].groupby("season"):
            if len(sub) < min_season_samples:
                print(f"  [group{group}] {season}: 표본 부족({len(sub)}) -> 스킵")
                continue
            all_rows.append(_corr_table(sub, target_col, group, f"유효구간-{season}"))

    result = pd.concat(all_rows, ignore_index=True)

    for scope in ["전체구간", "유효구간", "유효구간-봄", "유효구간-여름", "유효구간-가을", "유효구간-겨울"]:
        sub = result[(result["scope"] == scope) & (result["rank"] <= 5)]
        if sub.empty:
            continue
        print(f"\n=== {scope} (표본수: {sub.groupby('group')['n'].first().to_dict()}) ===")
        pivot = sub.pivot(index="rank", columns="group", values="grid_id")
        pivot_corr = sub.pivot(index="rank", columns="group", values="correlation").round(4)
        for r in pivot.index:
            print(f"  {r}위: " + " | ".join(
                f"g{g}: grid{int(pivot.loc[r, g])}({pivot_corr.loc[r, g]:.4f})" for g in pivot.columns
            ))

    result.to_csv("/home/khwak/wind-power/outputs/grid_correlation_extended.csv", index=False, encoding="utf-8-sig")
    print("\n📊 저장 완료: outputs/grid_correlation_extended.csv")
    return result


if __name__ == "__main__":
    run(mode="final")