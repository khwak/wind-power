"""
실제 프로덕션이 쓰는 격자(grid13, GRID_SELECTION_METHOD="manual" 기준)의
50m→117m 외삽 풍속과 SCADA 실측 풍속 잔차를, 시간 단위로 올바르게 정합해서 확인한다.
- SCADA는 10분 간격이므로 시간 평균으로 리샘플링 후 비교 (팀원 분석의 정시 단순매칭과 다름)
- cutoff 이전 구간만 사용
- 풍속 bin별 잔차/보정곡선 표까지 함께 출력 (비선형 보정 lookup 설계용)

실행: python diagnose_shear_bias_production.py
"""
import numpy as np
import pandas as pd

from config import RAW_DIR, GRID_SELECTION_METHOD, GRID_MANUAL_SELECTION
from prepare_data import (
    GROUP_META, _parse_times, _assert_one_issue_per_target, _resolve_cutoff,
    impute_ldaps_within_issue_cycle, add_ldaps_grid_features, get_kpx_group_centers,
    select_group_grid_id, _turbine_names, PREDICTION_REFERENCE_OFFSET_HOURS,
)

WS_BIN_WIDTH = 1.0


def load_data(mode="final", validation_start=None):
    train_labels_all = pd.read_csv(RAW_DIR / "train" / "train_labels.csv", encoding="utf-8-sig")
    ldaps_train = pd.read_csv(RAW_DIR / "train" / "ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv(RAW_DIR / "train" / "gfs_train.csv", encoding="utf-8-sig")
    ldaps_test = pd.read_csv(RAW_DIR / "test" / "ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv(RAW_DIR / "test" / "gfs_test.csv", encoding="utf-8-sig")
    scada_vestas = pd.read_csv(RAW_DIR / "train" / "scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv(RAW_DIR / "train" / "scada_unison_train.csv", encoding="utf-8-sig")

    _parse_times(train_labels_all, ldaps_train, gfs_train, ldaps_test, gfs_test, scada_vestas, scada_unison)
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

    scada_vestas_fit = scada_vestas[scada_vestas["kst_dtm"] < cutoff].copy()
    scada_unison_fit = scada_unison[scada_unison["kst_dtm"] < cutoff].copy()

    ldaps_train_grid = add_ldaps_grid_features(impute_ldaps_within_issue_cycle(ldaps_train))

    group_centers = get_kpx_group_centers(RAW_DIR / "info.xlsx")
    group_grids = {
        group: select_group_grid_id(
            group=group,
            center_lat=group_centers.loc[group, "lat"],
            center_lon=group_centers.loc[group, "lon"],
            ldaps_train_grid=ldaps_train_grid,
            train_labels_all=train_labels_all,
            target_col=f"kpx_group_{group}",
            cutoff=cutoff,
            method=GRID_SELECTION_METHOD,
            manual_selection=GRID_MANUAL_SELECTION,
        )
        for group in GROUP_META
    }
    print(f"실제 프로덕션 격자 선택: {group_grids}")

    return ldaps_train_grid, group_grids, scada_vestas_fit, scada_unison_fit, cutoff


def run(mode="final", validation_start=None):
    ldaps_train_grid, group_grids, scada_vestas_fit, scada_unison_fit, cutoff = load_data(mode, validation_start)

    all_rows = []
    for group, meta in GROUP_META.items():
        grid_id = group_grids[group]
        ldaps_g = ldaps_train_grid[
            (ldaps_train_grid["grid_id"] == grid_id)
            & (ldaps_train_grid["forecast_kst_dtm"] < cutoff)
        ][["forecast_kst_dtm", "ws117_power_ldaps", "alpha_shear_ldaps", "alpha_fallback_flag"]]

        scada_source = scada_vestas_fit if meta["maker"] == "vestas" else scada_unison_fit
        turbines = _turbine_names(meta["maker"], meta["turbines"])
        ws_cols = [f"{t}_ws" for t in turbines]

        # [핵심] 10분 -> 시간 평균 리샘플링 (팀원 분석의 정시 단순매칭과 차이나는 지점)
        scada_hourly = scada_source[["kst_dtm"] + ws_cols].copy()
        scada_hourly["forecast_kst_dtm"] = scada_hourly["kst_dtm"].dt.floor("h")
        scada_hourly["scada_ws_mean"] = scada_hourly[ws_cols].mean(axis=1)
        scada_hourly = scada_hourly.groupby("forecast_kst_dtm")["scada_ws_mean"].mean().reset_index()

        merged = ldaps_g.merge(scada_hourly, on="forecast_kst_dtm", how="inner").dropna(
            subset=["ws117_power_ldaps", "scada_ws_mean"]
        )
        merged["group"] = group
        merged["grid_id"] = grid_id
        merged["residual"] = merged["scada_ws_mean"] - merged["ws117_power_ldaps"]
        merged["ws_bin"] = np.floor(merged["ws117_power_ldaps"] / WS_BIN_WIDTH) * WS_BIN_WIDTH
        all_rows.append(merged)

    result = pd.concat(all_rows, ignore_index=True)

    print("\n===== 그룹별(=실제 사용 격자) 전체 잔차 통계 =====")
    print(result.groupby("group").agg(
        grid_id=("grid_id", "first"),
        n=("residual", "size"),
        mean_residual=("residual", "mean"),
        median_residual=("residual", "median"),
        mae=("residual", lambda s: s.abs().mean()),
        std_residual=("residual", "std"),
    ).round(3))

    print("\n===== alpha_fallback_flag별(외삽 실패 fallback=0.14 사용 여부) 잔차 비교 =====")
    print(result.groupby(["group", "alpha_fallback_flag"]).agg(
        n=("residual", "size"), mean_residual=("residual", "mean"), std_residual=("residual", "std"),
    ).round(3))

    print("\n===== 그룹별 풍속(LDAPS) 구간별 보정곡선용 표 (표본 30개 이상만) =====")
    for group in GROUP_META:
        sub = result[result["group"] == group]
        bin_summary = sub.groupby("ws_bin").agg(
            n=("residual", "size"),
            ldaps_ws_mean=("ws117_power_ldaps", "mean"),
            scada_ws_median=("scada_ws_mean", "median"),
            scada_ws_mean=("scada_ws_mean", "mean"),
            residual_median=("residual", "median"),
        ).reset_index()
        bin_summary = bin_summary[bin_summary["n"] >= 30]
        print(f"\n[group{group}, grid{sub['grid_id'].iloc[0]}]")
        print(bin_summary.round(3).to_string(index=False))

    out_path = "/home/khwak/wind-power/outputs/shear_bias_grid13_hourly.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n📊 저장 완료: {out_path}")
    return result


if __name__ == "__main__":
    run(mode="final")