"""
grid13(regime 축으로 채택된 격자)과 나머지 격자 간 차이(gradient)가,
grid13 하나만으로는 못 잡는 추가 정보를 갖고 있는지 편상관으로 확인한다.
cutoff 이전, 유효구간 기준 (기존 진단과 동일 제약).

실행: python diagnose_grid_gradient.py
"""
import numpy as np
import pandas as pd

from config import RAW_DIR, CAPACITY_KWH, VALID_RATIO_THRESHOLD
from prepare_data import (
    GROUP_META, _parse_times, _assert_one_issue_per_target, _resolve_cutoff,
    impute_ldaps_within_issue_cycle, add_ldaps_grid_features,
    PREDICTION_REFERENCE_OFFSET_HOURS,
)

PRIMARY_GRID = 13  # regime 축으로 채택된 격자


def _partial_corr(x, y, control):
    """control을 선형으로 제거한 뒤 x, y의 잔차끼리 상관관계(편상관)를 계산한다."""
    def resid(v):
        c = np.column_stack([control, np.ones(len(control))])
        coef, *_ = np.linalg.lstsq(c, v, rcond=None)
        return v - c @ coef
    return np.corrcoef(resid(x), resid(y))[0, 1]


def load_cutoff_eligible_wide(mode="final", validation_start=None):
    """cutoff 이전 데이터를 wide format(행=forecast_kst_dtm, 열=격자별 ws117)으로 만든다."""
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

    wide = fit_rows.pivot(index="forecast_kst_dtm", columns="grid_id", values="ws117_power_ldaps")
    wide.columns = [f"grid_{int(c)}" for c in wide.columns]
    return wide.reset_index(), train_labels_all


def run(mode="final", validation_start=None, primary_grid=PRIMARY_GRID):
    wide, train_labels_all = load_cutoff_eligible_wide(mode, validation_start)
    other_grids = [c for c in wide.columns if c.startswith("grid_") and c != f"grid_{primary_grid}"]

    all_rows = []
    for group in GROUP_META:
        target_col = f"kpx_group_{group}"
        capacity = CAPACITY_KWH[target_col]
        labels = train_labels_all[["kst_dtm", target_col]].rename(columns={"kst_dtm": "forecast_kst_dtm"})
        merged = wide.merge(labels, on="forecast_kst_dtm", how="inner").dropna(
            subset=[f"grid_{primary_grid}", target_col] + other_grids
        )
        merged = merged[merged[target_col] >= capacity * VALID_RATIO_THRESHOLD]  # 유효구간만

        y = merged[target_col].to_numpy()
        g_primary = merged[f"grid_{primary_grid}"].to_numpy()
        primary_corr = np.corrcoef(g_primary, y)[0, 1]
        print(f"[group{group}] grid{primary_grid} 단독 상관계수(유효구간): {primary_corr:.4f}")

        for other in other_grids:
            diff = g_primary - merged[other].to_numpy()
            all_rows.append({
                "group": group,
                "diff": f"grid{primary_grid}-{other.replace('grid_', '')}",
                "raw_corr_with_target": round(np.corrcoef(diff, y)[0, 1], 4),
                "partial_corr_given_primary": round(_partial_corr(diff, y, g_primary), 4),
                "n": len(merged),
            })

    result = pd.DataFrame(all_rows)
    result["abs_partial"] = result["partial_corr_given_primary"].abs()

    for group in GROUP_META:
        sub = result[result["group"] == group].sort_values("abs_partial", ascending=False).head(10)
        print(f"\n=== group{group}: grid{primary_grid} 통제 후 편상관 상위 10개 ===")
        print(sub[["diff", "raw_corr_with_target", "partial_corr_given_primary"]].to_string(index=False))

    result.to_csv("/home/khwak/wind-power/outputs/grid_gradient_partial_corr.csv", index=False, encoding="utf-8-sig")
    print("\n📊 저장 완료: outputs/grid_gradient_partial_corr.csv")
    return result


if __name__ == "__main__":
    run(mode="final")