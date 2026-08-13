from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import (
    RAW_DIR, TARGET_COLS, GRID_SELECTION_METHOD, GRID_MANUAL_SELECTION,
    SHEAR_BIAS_CORRECTION, GRID_DIFF_PAIRS, EXCLUDED_FEATURES,
    MODEL_DIR, SHEAR_MIN_SEASON_SAMPLES,
)

HUB_HEIGHT_M = 117.0
PREDICTION_REFERENCE_OFFSET_HOURS = 0

GROUP_META = {
    1: {
        "maker": "vestas",
        "turbines": list(range(1, 7)),
        "n_turbines": 6,
        "power_feature": "vestas_power_curve_pred_group1",
        "variability_feature": "vestas_power_variability_group1",
    },
    2: {
        "maker": "vestas",
        "turbines": list(range(7, 13)),
        "n_turbines": 6,
        "power_feature": "vestas_power_curve_pred_group2",
        "variability_feature": "vestas_power_variability_group2",
    },
    3: {
        "maker": "unison",
        "turbines": list(range(1, 6)),
        "n_turbines": 5,
        "power_feature": "unison_power_curve_pred",
        "variability_feature": "unison_power_variability",
    },
}


# -----------------------------------------------------------------------------
# 공통 유틸
# -----------------------------------------------------------------------------
def _parse_times(*frames: pd.DataFrame) -> None:
    """존재하는 시간 컬럼을 모두 datetime으로 변환한다."""
    for frame in frames:
        for col in ("forecast_kst_dtm", "data_available_kst_dtm", "kst_dtm"):
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col])


def calendar_features(dt_series: pd.Series) -> pd.DataFrame:
    """예측 대상 시각에서 직접 계산 가능한 달력 피처."""
    dt = pd.to_datetime(dt_series)
    out = pd.DataFrame(index=dt.index)

    hour = dt.dt.hour
    day_of_year = dt.dt.dayofyear

    out["month"] = dt.dt.month.astype("int8")
    out["hour"] = hour.astype("int8")
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype("int8")
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dayofyear_sin"] = np.sin(2 * np.pi * (day_of_year - 1) / 365.25)
    out["dayofyear_cos"] = np.cos(2 * np.pi * (day_of_year - 1) / 365.25)
    return out


def dms_to_decimal(dms_str: str) -> tuple[float, float]:
    """DMS 좌표 문자열을 (위도, 경도) 십진수 좌표로 변환한다."""
    if not isinstance(dms_str, str):
        raise TypeError(f"좌표가 문자열이 아닙니다: {dms_str!r}")

    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*°\s*(\d+(?:\.\d+)?)\s*['′]\s*"
        r"(\d+(?:\.\d+)?)\s*[\"″]\s*([NSEW])",
        flags=re.IGNORECASE,
    )
    matches = pattern.findall(dms_str.strip())
    if len(matches) != 2:
        raise ValueError(f"DMS 좌표 형식을 해석할 수 없습니다: {dms_str!r}")

    def convert(parts: tuple[str, str, str, str]) -> float:
        deg, minute, sec, direction = parts
        value = float(deg) + float(minute) / 60.0 + float(sec) / 3600.0
        if direction.upper() in {"S", "W"}:
            value *= -1
        return value

    return convert(matches[0]), convert(matches[1])


def get_kpx_group_centers(info_xlsx_path: Path) -> pd.DataFrame:
    """info.xlsx에서 KPX 그룹별 터빈 중심 좌표를 계산한다."""
    info = pd.read_excel(info_xlsx_path, sheet_name="info", header=3)
    info = info.loc[:, ~info.columns.astype(str).str.startswith("Unnamed")]
    info = info[info["호기"].notna()].copy()
    info["KPX그룹"] = info["KPX그룹"].ffill().astype(int)

    coords = info["좌표(Google)"].map(dms_to_decimal)
    info["lat"] = coords.map(lambda value: value[0])
    info["lon"] = coords.map(lambda value: value[1])
    return info.groupby("KPX그룹")[["lat", "lon"]].mean()


def _haversine_km(lat1, lon1, lat2, lon2):
    """위경도 사이 대권거리(km)."""
    radius = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * radius * np.arcsin(np.sqrt(a))


def nearest_ldaps_grid_id(
    center_lat: float,
    center_lon: float,
    ldaps_df: pd.DataFrame,
) -> int:
    """그룹 중심과 가장 가까운 LDAPS 격자를 찾는다."""
    grids = (
        ldaps_df[["grid_id", "latitude", "longitude"]]
        .drop_duplicates("grid_id")
        .copy()
    )
    grids["distance_km"] = _haversine_km(
        center_lat,
        center_lon,
        grids["latitude"].to_numpy(),
        grids["longitude"].to_numpy(),
    )
    return int(grids.loc[grids["distance_km"].idxmin(), "grid_id"])

def select_group_grid_id(
    group: int,
    center_lat: float,
    center_lon: float,
    ldaps_train_grid: pd.DataFrame,
    train_labels_all: pd.DataFrame,
    target_col: str,
    cutoff: pd.Timestamp,
    method: str = "nearest",
    manual_selection: dict[int, int] | None = None,
) -> int:
    if method == "nearest":
        return nearest_ldaps_grid_id(center_lat, center_lon, ldaps_train_grid)

    if method == "manual":
        if manual_selection is None or group not in manual_selection:
            raise ValueError(f"GRID_MANUAL_SELECTION에 그룹 {group} 값이 없습니다.")
        return manual_selection[group]
    
    if method == "correlation":
        fit_rows = ldaps_train_grid[ldaps_train_grid["forecast_kst_dtm"] < cutoff]
        labels = train_labels_all[["kst_dtm", target_col]].rename(
            columns={"kst_dtm": "forecast_kst_dtm"}
        )
        merged = fit_rows.merge(labels, on="forecast_kst_dtm", how="inner").dropna(
            subset=["ws117_power_ldaps", target_col]
        )
        if merged.empty:
            raise ValueError(f"그룹 {group}: 상관관계 계산할 표본이 없습니다.")

        corr_by_grid = (
            merged.groupby("grid_id")
            .apply(lambda g: g["ws117_power_ldaps"].corr(g[target_col]))
            .dropna()
        )
        if corr_by_grid.empty:
            raise ValueError(f"그룹 {group}: 유효한 상관관계가 없습니다.")
        return int(corr_by_grid.idxmax())

    raise ValueError(f"알 수 없는 grid 선택 방식: {method}")

def diagnose_grid_correlation(
    ldaps_train_grid: pd.DataFrame,
    train_labels_all: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """그룹별로 16개 격자 전부의 상관계수를 보여준다. 1등과 2등 차이가 작으면
    correlation 방식의 '승자'가 사실상 노이즈일 가능성이 높다는 뜻이다."""
    rows = []
    for group in GROUP_META:
        target_col = f"kpx_group_{group}"
        fit_rows = ldaps_train_grid[ldaps_train_grid["forecast_kst_dtm"] < cutoff]
        labels = train_labels_all[["kst_dtm", target_col]].rename(columns={"kst_dtm": "forecast_kst_dtm"})
        merged = fit_rows.merge(labels, on="forecast_kst_dtm", how="inner").dropna(
            subset=["ws117_power_ldaps", target_col]
        )
        corr = merged.groupby("grid_id").apply(
            lambda g: g["ws117_power_ldaps"].corr(g[target_col]),
            include_groups=False
        ).sort_values(ascending=False)
        corr_df = corr.reset_index()
        corr_df.columns = ["grid_id", "correlation"]
        corr_df["group"] = group
        corr_df["rank"] = range(1, len(corr_df) + 1)
        rows.append(corr_df)

    result = pd.concat(rows, ignore_index=True)
    print(result[result["rank"] <= 5].to_string(index=False))
    return result

def diagnose_shear_bias(
    ldaps_train_grid: pd.DataFrame,
    group_grids: dict[int, int],
    scada_vestas_fit: pd.DataFrame,
    scada_unison_fit: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """
    그룹별 대표 격자의 외삽 117m 풍속과 SCADA 실측 풍속을 시간 단위로 매칭해 잔차를 계산한다.
    cutoff 이전만 사용한다. 팀원 shear 분석에 그대로 쓸 수 있고, 지금 바로 셀프 진단도 가능하다.
    """
    rows = []
    for group, meta in GROUP_META.items():
        grid_id = group_grids[group]
        ldaps_g = ldaps_train_grid[
            (ldaps_train_grid["grid_id"] == grid_id)
            & (ldaps_train_grid["forecast_kst_dtm"] < cutoff)
        ][["forecast_kst_dtm", "ws117_power_ldaps", "alpha_fallback_flag"]]

        scada_source = scada_vestas_fit if meta["maker"] == "vestas" else scada_unison_fit
        turbines = _turbine_names(meta["maker"], meta["turbines"])
        ws_cols = [f"{t}_ws" for t in turbines]

        scada_hourly = scada_source[["kst_dtm"] + ws_cols].copy()
        scada_hourly["forecast_kst_dtm"] = scada_hourly["kst_dtm"].dt.floor("h")
        scada_hourly["scada_ws_mean"] = scada_hourly[ws_cols].mean(axis=1)
        scada_hourly = scada_hourly.groupby("forecast_kst_dtm")["scada_ws_mean"].mean().reset_index()

        merged = ldaps_g.merge(scada_hourly, on="forecast_kst_dtm", how="inner")
        merged["group"] = group
        merged["residual"] = merged["ws117_power_ldaps"] - merged["scada_ws_mean"]
        rows.append(merged)

    result = pd.concat(rows, ignore_index=True)
    print(result.groupby(["group", "alpha_fallback_flag"])["residual"].agg(["count", "mean", "std"]).round(3))
    return result


def apply_shear_bias_correction(ws117, group: int) -> np.ndarray:
    """SHEAR_BIAS_CORRECTION에 그룹별 보정값이 있으면 선형 보정을 적용한다.
    없으면(기본) 원본을 그대로 반환 — 기존 동작과 100% 동일."""
    params = SHEAR_BIAS_CORRECTION.get(group)
    if not params:
        return np.asarray(ws117, dtype=float)
    scale = params.get("scale", 1.0)
    offset = params.get("offset", 0.0)
    return np.asarray(ws117, dtype=float) * scale + offset

def _turbine_names(maker: str, turbine_numbers: Iterable[int]) -> list[str]:
    return [f"{maker}_wtg{i:02d}" for i in turbine_numbers]


# -----------------------------------------------------------------------------
# 예보 가용시각과 cutoff
# -----------------------------------------------------------------------------
def _assert_one_issue_per_target(df: pd.DataFrame, source_name: str) -> None:
    """한 예보 대상 시각에 복수의 사용 가능 시각이 섞였는지 검사한다."""
    required = {"forecast_kst_dtm", "data_available_kst_dtm"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{source_name}에 필수 컬럼이 없습니다: {sorted(missing)}")

    count = df.groupby("forecast_kst_dtm")["data_available_kst_dtm"].nunique()
    if not count.eq(1).all():
        bad_times = count[count.ne(1)].index[:5].tolist()
        raise ValueError(
            f"{source_name}: 같은 forecast_kst_dtm에 복수 예보 발행본이 섞였습니다. "
            f"예: {bad_times}. data_available_kst_dtm 기준으로 먼저 선택해야 합니다."
        )


def _first_available_for_target_period(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    first_target: pd.Timestamp,
) -> pd.Timestamp:
    """첫 예측 대상 시각에 대응하는 가장 보수적인 예보 사용 가능 시각."""
    candidates: list[pd.Timestamp] = []

    for frame, name in ((ldaps, "LDAPS"), (gfs, "GFS")):
        target_rows = frame[frame["forecast_kst_dtm"] == first_target]
        if target_rows.empty:
            raise ValueError(f"{name}에 첫 대상 시각 {first_target} 행이 없습니다.")
        available_values = target_rows["data_available_kst_dtm"].dropna().unique()
        if len(available_values) != 1:
            raise ValueError(
                f"{name}의 첫 대상 시각 {first_target}에 사용 가능 시각이 "
                f"정확히 1개가 아닙니다: {available_values}"
            )
        candidates.append(pd.Timestamp(available_values[0]))

    # 두 소스가 다르면 더 이른 시각을 cutoff로 잡는 것이 실제 관측자료 사용에는 보수적이다.
    return min(candidates)


def _resolve_cutoff(
    mode: str,
    validation_start: str | pd.Timestamp | None,
    ldaps_train: pd.DataFrame,
    gfs_train: pd.DataFrame,
    ldaps_test: pd.DataFrame,
    gfs_test: pd.DataFrame,
    prediction_reference_offset_hours: int,
) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    """최종 제출 또는 시간 검증에 사용할 관측자료 cutoff를 계산한다."""
    if mode not in {"final", "validation"}:
        raise ValueError("mode는 'final' 또는 'validation'이어야 합니다.")

    if mode == "final":
        first_target = min(
            ldaps_test["forecast_kst_dtm"].min(),
            gfs_test["forecast_kst_dtm"].min(),
        )
        available = _first_available_for_target_period(
            ldaps_test,
            gfs_test,
            first_target,
        )
        cutoff = available + pd.Timedelta(hours=prediction_reference_offset_hours)
        return cutoff, None

    if validation_start is None:
        raise ValueError("mode='validation'이면 validation_start가 필요합니다.")

    requested_start = pd.Timestamp(validation_start)
    candidate_targets = sorted(
        set(ldaps_train.loc[
            ldaps_train["forecast_kst_dtm"] >= requested_start,
            "forecast_kst_dtm",
        ]).intersection(
            set(gfs_train.loc[
                gfs_train["forecast_kst_dtm"] >= requested_start,
                "forecast_kst_dtm",
            ])
        )
    )
    if not candidate_targets:
        raise ValueError(f"validation_start={requested_start} 이후 공통 예보 시각이 없습니다.")

    actual_validation_start = pd.Timestamp(candidate_targets[0])
    available = _first_available_for_target_period(
        ldaps_train,
        gfs_train,
        actual_validation_start,
    )
    cutoff = available + pd.Timedelta(hours=prediction_reference_offset_hours)
    return cutoff, actual_validation_start


def _filter_observed_before(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """실측/라벨은 집계 종료 시각이 cutoff보다 엄격히 이전인 행만 허용한다."""
    if "kst_dtm" not in df.columns:
        raise KeyError("실측 데이터에 kst_dtm이 없습니다.")
    filtered = df[df["kst_dtm"] < cutoff].copy()
    if filtered.empty:
        raise ValueError(f"cutoff={cutoff} 이전 실측 데이터가 없습니다.")
    return filtered


# -----------------------------------------------------------------------------
# LDAPS 결측 처리와 파생변수
# -----------------------------------------------------------------------------
def impute_ldaps_within_issue_cycle(ldaps: pd.DataFrame) -> pd.DataFrame:
    """
    결측값을 같은 grid_id·같은 data_available_kst_dtm 묶음 안에서만 보간한다.

    같은 예보 발행 묶음의 24시간 예보는 동시에 공개되므로, 그 묶음 안에서
    forecast 대상 시각 앞뒤 값을 이용하는 것은 사후 관측값 사용이 아니다.
    TRAIN과 TEST는 절대 합쳐 보간하지 않는다.
    """
    out = ldaps.copy()
    out = out.sort_values(
        ["grid_id", "data_available_kst_dtm", "forecast_kst_dtm"]
    ).reset_index(drop=True)

    id_cols = {
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        "latitude",
        "longitude",
    }
    numeric_cols = [
        col for col in out.columns
        if col not in id_cols and pd.api.types.is_numeric_dtype(out[col])
    ]

    out["ldaps_missing_flag"] = out[numeric_cols].isna().any(axis=1).astype("int8")

    static_cols = [
        col for col in ("surface_0_lsm", "surface_0_h")
        if col in out.columns
    ]
    dynamic_cols = [col for col in numeric_cols if col not in static_cols]

    for col in static_cols:
        out[col] = out.groupby("grid_id", sort=False)[col].transform(
            lambda series: series.ffill().bfill()
        )

    if dynamic_cols:
        out[dynamic_cols] = (
            out.groupby(
                ["grid_id", "data_available_kst_dtm"],
                sort=False,
            )[dynamic_cols]
            .transform(
                lambda frame: frame.interpolate(
                    method="linear",
                    limit_direction="both",
                )
            )
        )

    remaining = out[dynamic_cols].isna().sum() if dynamic_cols else pd.Series(dtype=int)
    if remaining.sum() > 0:
        bad = remaining[remaining > 0].to_dict()
        raise ValueError(
            "같은 예보 발행 묶음 안에서 해결되지 않은 LDAPS 결측이 있습니다. "
            f"다른 발행시각 자료로 채우지 말고 원인을 확인하세요: {bad}"
        )
    return out


def add_ldaps_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    """한 행 내부의 예보값만으로 117m 풍속·풍향·밀도·에너지 피처를 생성한다."""
    out = df.copy()

    u10 = out["heightAboveGround_10_10u"]
    v10 = out["heightAboveGround_10_10v"]
    ws10 = np.hypot(u10, v10)

    u50 = (
        out["heightAboveGround_50_50MUmax"]
        + out["heightAboveGround_50_50MUmin"]
    ) / 2.0
    v50 = (
        out["heightAboveGround_50_50MVmax"]
        + out["heightAboveGround_50_50MVmin"]
    ) / 2.0
    ws50 = np.hypot(u50, v50)

    valid = (ws10 >= 0.5) & (ws50 >= 0.5)
    alpha_raw = pd.Series(np.nan, index=out.index, dtype=float)
    alpha_raw.loc[valid] = (
        np.log(ws50.loc[valid] / ws10.loc[valid]) / np.log(50.0 / 10.0)
    )

    # 데이터 전체 분위수로 clipping 범위를 fit하지 않고 고정 물리 범위를 사용한다.
    alpha = alpha_raw.replace([np.inf, -np.inf], np.nan).fillna(0.14).clip(-0.5, 0.6)
    scale = (HUB_HEIGHT_M / 50.0) ** alpha
    u117 = u50 * scale
    v117 = v50 * scale

    out["alpha_shear_ldaps"] = alpha
    out["alpha_fallback_flag"] = (~valid).astype("int8")
    out["ws117_power_ldaps"] = np.hypot(u117, v117)
    out["wd117_ldaps"] = np.degrees(np.arctan2(-u117, -v117)) % 360

    temperature_k = out["heightAboveGround_2_t"].clip(lower=180.0)
    specific_humidity = out["heightAboveGround_2_q"].clip(0.0, 0.05)
    virtual_temperature = temperature_k * (1.0 + 0.61 * specific_humidity)
    out["air_density_ldaps"] = out["surface_0_sp"] / (
        287.058 * virtual_temperature
    )
    out["wind_energy_flux_grid_ldaps"] = (
        0.5 * out["air_density_ldaps"] * out["ws117_power_ldaps"] ** 3
    )
    return out


def _pivot_grid_feature(
    df: pd.DataFrame,
    value_col: str,
    output_prefix: str,
) -> pd.DataFrame:
    wide = df.pivot(
        index="forecast_kst_dtm",
        columns="grid_id",
        values=value_col,
    )
    wide.columns = [f"{output_prefix}_{int(grid_id)}" for grid_id in wide.columns]
    return wide.reset_index()

def add_grid_diff_features(df: pd.DataFrame, pairs: list[tuple[int, int]]) -> pd.DataFrame:
    """지정된 격자 쌍의 ws117 차이(gradient)를 새 feature로 추가한다.
    diff = ws117_ldaps_grid_a - ws117_ldaps_grid_b.
    grid13 하나만으로는 못 잡는 공간적 정보를 편상관으로 확인한 뒤 반영하는 feature다."""
    out = df.copy()
    for a, b in pairs:
        col_a, col_b = f"ws117_ldaps_grid_{a}", f"ws117_ldaps_grid_{b}"
        if col_a not in out.columns or col_b not in out.columns:
            raise KeyError(f"격자 {a} 또는 {b}의 ws117 피처가 없습니다: {col_a}, {col_b}")
        out[f"ws117_ldaps_diff_{a}_{b}"] = out[col_a] - out[col_b]
    return out

def build_ldaps_hourly_pair(
    ldaps_train_grid: pd.DataFrame,
    ldaps_test_grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    LDAPS를 시간당 1행으로 만들고 인과 방향 시계열 피처를 계산한다.

    TRAIN과 TEST를 이어 계산하는 것은 TEST 첫 구간의 24시간 lag를 TRAIN의
    과거 예보로 연결하기 위한 것이다. 미래 방향 연산은 사용하지 않는다.
    """
    train = ldaps_train_grid.copy()
    test = ldaps_test_grid.copy()
    train["_split"] = "train"
    test["_split"] = "test"
    all_grid = pd.concat([train, test], ignore_index=True)

    _assert_one_issue_per_target(all_grid, "LDAPS")

    ws_wide = _pivot_grid_feature(
        all_grid,
        "ws117_power_ldaps",
        "ws117_ldaps_grid",
    )
    wd_wide = _pivot_grid_feature(
        all_grid,
        "wd117_ldaps",
        "wd117_ldaps_grid",
    )

    spatial = all_grid.groupby("forecast_kst_dtm", as_index=False).agg(
        ws117_ldaps_spatial_mean=("ws117_power_ldaps", "mean"),
        ws117_ldaps_spatial_std=("ws117_power_ldaps", "std"),
        ws117_ldaps_spatial_min=("ws117_power_ldaps", "min"),
        ws117_ldaps_spatial_max=("ws117_power_ldaps", "max"),
        alpha_shear_ldaps=("alpha_shear_ldaps", "mean"),
        air_density_ldaps=("air_density_ldaps", "mean"),
        prmsl_range_ldaps=("meanSea_0_prmsl", lambda values: values.max() - values.min()),
        wind_energy_flux_ldaps=("wind_energy_flux_grid_ldaps", "mean"),
        ldaps_missing_flag=("ldaps_missing_flag", "max"),
        alpha_fallback_fraction=("alpha_fallback_flag", "mean"),
        ldaps_data_available_kst_dtm=("data_available_kst_dtm", "first"),
        _split=("_split", "first"),
    )

    hourly = (
        ws_wide
        .merge(wd_wide, on="forecast_kst_dtm", how="left", validate="one_to_one")
        .merge(spatial, on="forecast_kst_dtm", how="left", validate="one_to_one")
        .sort_values("forecast_kst_dtm")
        .reset_index(drop=True)
    )

    hourly = add_grid_diff_features(hourly, GRID_DIFF_PAIRS)  # 격자 간 gradient feature
    hourly["ws117_cube_ldaps"] = hourly["ws117_ldaps_spatial_mean"] ** 3
    hourly["ldaps_lead_hour"] = (
        hourly["forecast_kst_dtm"]
        - pd.to_datetime(hourly["ldaps_data_available_kst_dtm"])
    ).dt.total_seconds() / 3600.0

    # target 시간이 증가할 때 예보 사용 가능 시각이 뒤로 돌아가면 lag 안전성을 재검토해야 한다.
    if not hourly["ldaps_data_available_kst_dtm"].is_monotonic_increasing:
        raise ValueError(
            "LDAPS data_available_kst_dtm이 target 시간순으로 단조 증가하지 않습니다."
        )

    base = hourly["ws117_ldaps_spatial_mean"]
    hourly["ws117_diff_1h_ldaps"] = base.diff(1)
    hourly["ws117_lag_24h_ldaps"] = base.shift(24)
    hourly["ws117_roll_mean_6h_ldaps"] = base.rolling(
        window=6,
        min_periods=1,
        center=False,
    ).mean()
    hourly["ws117_roll_std_6h_ldaps"] = base.rolling(
        window=6,
        min_periods=2,
        center=False,
    ).std()

    train_out = (
        hourly[hourly["_split"] == "train"]
        .drop(columns="_split")
        .reset_index(drop=True)
    )
    test_out = (
        hourly[hourly["_split"] == "test"]
        .drop(columns="_split")
        .reset_index(drop=True)
    )
    return train_out, test_out


# -----------------------------------------------------------------------------
# GFS 피처와 공간 집계
# -----------------------------------------------------------------------------
def add_gfs_features(df: pd.DataFrame) -> pd.DataFrame:
    """GFS 한 행 내부의 예보값만 사용하는 안전한 파생변수."""
    out = df.copy()
    ws10 = np.hypot(
        out["heightAboveGround_10_10u"],
        out["heightAboveGround_10_10v"],
    )
    out["ws10_gfs"] = ws10
    out["gust_excess_gfs"] = (out["surface_0_gust"] - ws10).clip(lower=0)
    out["calm_wind_flag_gfs"] = (ws10 < 0.5).astype("int8")
    out["gust_factor_proxy_gfs"] = (
        out["surface_0_gust"] / ws10.clip(lower=0.5)
    ).clip(0, 5)
    return out


def aggregate_weather(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    격자별 예보를 시간당 평균으로 집계한다.

    원본과 달리 data_available_kst_dtm을 버리지 않고 보존하며,
    같은 target 시각에 복수 발행본이 섞였는지 먼저 검사한다.
    """
    frame = df.copy()
    _assert_one_issue_per_target(frame, prefix.upper())

    excluded = {
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        "latitude",
        "longitude",
    }
    value_cols = [
        col for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]

    mean_values = frame.groupby("forecast_kst_dtm")[value_cols].mean()
    mean_values.columns = [f"{prefix}_{col}_mean" for col in mean_values.columns]

    available = (
        frame.groupby("forecast_kst_dtm")["data_available_kst_dtm"]
        .first()
        .rename(f"{prefix}_data_available_kst_dtm")
    )

    out = mean_values.join(available).reset_index()
    out[f"{prefix}_lead_hour"] = (
        out["forecast_kst_dtm"]
        - pd.to_datetime(out[f"{prefix}_data_available_kst_dtm"])
    ).dt.total_seconds() / 3600.0
    return out


# -----------------------------------------------------------------------------
# SCADA 룩업: 반드시 cutoff 이전 자료만 fit
# -----------------------------------------------------------------------------
def fit_power_curve_lookup(
    scada_fit: pd.DataFrame,
    turbines: list[str],
    power_suffix: str = "_power_kw10m",
    ws_suffix: str = "_ws",
    ws_bin_width: float = 0.5,
    min_count: int = 30,
) -> pd.DataFrame:
    """cutoff 이전 SCADA로 풍속-bin별 터빈 1기 발전량 중앙값을 학습한다."""
    pooled_parts = []
    for turbine in turbines:
        pooled_parts.append(
            pd.DataFrame({
                "power": scada_fit[f"{turbine}{power_suffix}"],
                "ws": scada_fit[f"{turbine}{ws_suffix}"],
            })
        )

    pooled = pd.concat(pooled_parts, ignore_index=True).dropna()
    pooled = pooled[np.isfinite(pooled["power"]) & np.isfinite(pooled["ws"])].copy()
    pooled["power"] = pooled["power"].clip(lower=0)
    pooled["ws_bin"] = np.floor(pooled["ws"] / ws_bin_width)

    lookup = (
        pooled.groupby("ws_bin")["power"]
        .agg(median="median", count="count")
        .reset_index()
        .sort_values("ws_bin")
    )
    lookup.loc[lookup["count"] < min_count, "median"] = np.nan
    lookup["median"] = lookup["median"].interpolate(limit_direction="both")
    lookup["ws_center"] = (lookup["ws_bin"] + 0.5) * ws_bin_width

    lookup = lookup.dropna(subset=["median", "ws_center"])
    if lookup.empty:
        raise ValueError("유효한 SCADA 파워커브 룩업을 만들 수 없습니다.")
    return lookup[["ws_center", "median", "count"]]

def fit_shear_calibration_lookup(
    ldaps_ws,
    scada_ws,
    ws_bin_width: float = 1.0,
    min_count: int = 30,
) -> pd.DataFrame:
    """cutoff 이전, 같은 그룹의 (LDAPS 117m 외삽풍속, SCADA 실측풍속) 쌍으로
    LDAPS 풍속 -> 보정된(SCADA 등가) 풍속 룩업을 학습한다. 비선형 관계를 그대로 반영한다."""
    pooled = pd.DataFrame({"ldaps": ldaps_ws, "scada": scada_ws}).dropna()
    pooled = pooled[np.isfinite(pooled["ldaps"]) & np.isfinite(pooled["scada"])].copy()
    pooled["ws_bin"] = np.floor(pooled["ldaps"] / ws_bin_width)

    lookup = (
        pooled.groupby("ws_bin")["scada"]
        .agg(median="median", count="count")
        .reset_index()
        .sort_values("ws_bin")
    )
    lookup.loc[lookup["count"] < min_count, "median"] = np.nan
    lookup["median"] = lookup["median"].interpolate(limit_direction="both")
    lookup["ldaps_center"] = (lookup["ws_bin"] + 0.5) * ws_bin_width

    lookup = lookup.dropna(subset=["median", "ldaps_center"])
    if lookup.empty:
        raise ValueError("유효한 shear 보정 룩업을 만들 수 없습니다.")
    return lookup[["ldaps_center", "median", "count"]]


def apply_shear_calibration_lookup(wind_speed, lookup: pd.DataFrame) -> np.ndarray:
    """LDAPS 풍속을 보정 룩업으로 SCADA-등가 풍속으로 치환한다."""
    ws = np.asarray(wind_speed, dtype=float)
    result = np.full(len(ws), np.nan, dtype=float)
    valid = np.isfinite(ws)
    result[valid] = np.interp(
        ws[valid],
        lookup["ldaps_center"].to_numpy(),
        lookup["median"].to_numpy(),
    )
    return result


def fit_group_shear_calibration(
    group: int,
    meta: dict,
    ldaps_train_grid: pd.DataFrame,
    grid_id: int,
    scada_vestas_fit: pd.DataFrame,
    scada_unison_fit: pd.DataFrame,
    cutoff: pd.Timestamp,
    months: list[int] | None = None,  # [신규] 계절별 보정용 월 필터
) -> pd.DataFrame:
    """cutoff 이전 구간에서 그룹 대표 격자의 ws117과 SCADA 실측(시간평균)을 매칭해
    shear 보정 룩업을 학습한다. months가 주어지면 그 달들만 필터링한다."""
    ldaps_g = ldaps_train_grid[
        (ldaps_train_grid["grid_id"] == grid_id)
        & (ldaps_train_grid["forecast_kst_dtm"] < cutoff)
    ][["forecast_kst_dtm", "ws117_power_ldaps"]]

    scada_source = scada_vestas_fit if meta["maker"] == "vestas" else scada_unison_fit
    turbines = _turbine_names(meta["maker"], meta["turbines"])
    ws_cols = [f"{t}_ws" for t in turbines]

    scada_hourly = scada_source[["kst_dtm"] + ws_cols].copy()
    scada_hourly["forecast_kst_dtm"] = scada_hourly["kst_dtm"].dt.floor("h")
    scada_hourly["scada_ws_mean"] = scada_hourly[ws_cols].mean(axis=1)
    scada_hourly = scada_hourly.groupby("forecast_kst_dtm")["scada_ws_mean"].mean().reset_index()

    merged = ldaps_g.merge(scada_hourly, on="forecast_kst_dtm", how="inner").dropna()
    if months is not None:  # [신규]
        merged = merged[merged["forecast_kst_dtm"].dt.month.isin(months)]
    return fit_shear_calibration_lookup(merged["ws117_power_ldaps"], merged["scada_ws_mean"])


def fit_group_shear_calibration_seasonal(
    group: int,
    meta: dict,
    ldaps_train_grid: pd.DataFrame,
    grid_id: int,
    scada_vestas_fit: pd.DataFrame,
    scada_unison_fit: pd.DataFrame,
    cutoff: pd.Timestamp,
    season_months: dict[str, list[int]],
    min_season_samples: int = SHEAR_MIN_SEASON_SAMPLES,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """계절 클러스터(regime 발견에 쓰던 seasonal_clusters.json과 동일 클러스터)별로
    shear 보정 룩업을 따로 학습한다. 표본 부족한 계절은 그룹 전체(글로벌) 룩업으로 자동 폴백한다."""
    global_lookup = fit_group_shear_calibration(
        group, meta, ldaps_train_grid, grid_id, scada_vestas_fit, scada_unison_fit, cutoff,
    )
    season_lookups: dict[str, pd.DataFrame] = {}
    for season_name, months in season_months.items():
        try:
            lk = fit_group_shear_calibration(
                group, meta, ldaps_train_grid, grid_id, scada_vestas_fit, scada_unison_fit, cutoff,
                months=months,
            )
            total_count = int(lk["count"].sum())
            if total_count < min_season_samples:
                print(f"  [group{group}] {season_name}{months} shear 보정 표본 부족({total_count}) -> 글로벌 폴백")
                season_lookups[season_name] = global_lookup
            else:
                season_lookups[season_name] = lk
        except ValueError:
            print(f"  [group{group}] {season_name}{months} shear 보정 실패 -> 글로벌 폴백")
            season_lookups[season_name] = global_lookup
    return season_lookups, global_lookup


def apply_shear_calibration_seasonal(
    wind_speed,
    forecast_kst_dtm,
    season_lookups: dict[str, pd.DataFrame],
    global_lookup: pd.DataFrame,
    group_clusters: dict[str, list[int]],
) -> np.ndarray:
    """행의 월(forecast_kst_dtm)로 계절을 찾아 해당 계절 전용 룩업으로 보정한다.
    어느 클러스터에도 안 걸리는 달은 글로벌 룩업으로 처리한다."""
    ws = np.asarray(wind_speed, dtype=float)
    months = pd.to_datetime(forecast_kst_dtm).dt.month.to_numpy()
    result = np.full(len(ws), np.nan, dtype=float)
    assigned = np.zeros(len(ws), dtype=bool)

    for season_name, season_months in group_clusters.items():
        mask = np.isin(months, season_months) & (~assigned)
        if not mask.any():
            continue
        lookup = season_lookups.get(season_name, global_lookup)
        result[mask] = apply_shear_calibration_lookup(ws[mask], lookup)
        assigned |= mask

    if (~assigned).any():
        result[~assigned] = apply_shear_calibration_lookup(ws[~assigned], global_lookup)
    return result


def apply_power_curve_lookup(
    wind_speed,
    lookup: pd.DataFrame,
    n_turbines: int,
) -> np.ndarray:
    """예보 풍속으로 경험적 파워커브를 조회한다."""
    ws = np.asarray(wind_speed, dtype=float)
    result = np.full(len(ws), np.nan, dtype=float)
    valid = np.isfinite(ws)
    result[valid] = np.interp(
        ws[valid],
        lookup["ws_center"].to_numpy(),
        lookup["median"].to_numpy(),
    ) * n_turbines
    return result


def fit_variability_lookup(
    scada_fit: pd.DataFrame,
    turbines: list[str],
    power_suffix: str = "_power_kw10m",
    ws_suffix: str = "_ws",
    wd_suffix: str = "_wd",
    ws_bin_width: float = 0.5,
    wd_bin_width: int = 30,
    min_count: int = 30,
) -> pd.DataFrame:
    """cutoff 이전 SCADA로 풍속·풍향 조건별 발전량 변동성을 학습한다."""
    pooled_parts = []
    for turbine in turbines:
        pooled_parts.append(
            pd.DataFrame({
                "power": scada_fit[f"{turbine}{power_suffix}"],
                "ws": scada_fit[f"{turbine}{ws_suffix}"],
                "wd": scada_fit[f"{turbine}{wd_suffix}"] % 360,
            })
        )

    pooled = pd.concat(pooled_parts, ignore_index=True).dropna()
    pooled["power"] = pooled["power"].clip(lower=0)
    pooled["ws_bin"] = np.floor(pooled["ws"] / ws_bin_width)
    pooled["wd_bin"] = np.floor(pooled["wd"] / wd_bin_width)

    lookup = (
        pooled.groupby(["ws_bin", "wd_bin"])["power"]
        .agg(std="std", count="count")
        .reset_index()
    )
    lookup.loc[lookup["count"] < min_count, "std"] = np.nan
    if lookup["std"].notna().sum() == 0:
        raise ValueError("유효한 SCADA 변동성 룩업을 만들 수 없습니다.")
    return lookup[["ws_bin", "wd_bin", "std", "count"]]


def apply_variability_lookup(
    wind_speed,
    wind_direction,
    lookup: pd.DataFrame,
    ws_bin_width: float = 0.5,
    wd_bin_width: int = 30,
) -> np.ndarray:
    """행 순서를 보존하면서 조건별 변동성 값을 조회한다."""
    ws = np.asarray(wind_speed, dtype=float)
    wd = np.asarray(wind_direction, dtype=float)

    query = pd.DataFrame({
        "_row_id": np.arange(len(ws)),
        "ws_bin": np.floor(ws / ws_bin_width),
        "wd_bin": np.floor((wd % 360) / wd_bin_width),
    })

    merged = query.merge(
        lookup[["ws_bin", "wd_bin", "std"]],
        on=["ws_bin", "wd_bin"],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    speed_fallback = (
        lookup.groupby("ws_bin")["std"]
        .median()
        .rename("std_speed_fallback")
    )
    merged = merged.merge(
        speed_fallback,
        on="ws_bin",
        how="left",
        sort=False,
        validate="many_to_one",
    )

    global_fallback = float(lookup["std"].median())
    merged["result"] = (
        merged["std"]
        .fillna(merged["std_speed_fallback"])
        .fillna(global_fallback)
    )
    return merged.sort_values("_row_id")["result"].to_numpy()


def _apply_scada_proxy_features(
    frame: pd.DataFrame,
    group_grids: dict[int, int],
    power_lookups: dict[int, pd.DataFrame],
    variability_lookups: dict[int, pd.DataFrame],
    shear_season_lookups: dict[int, tuple[dict[str, pd.DataFrame], pd.DataFrame]], 
    seasonal_clusters: dict[str, dict[str, list[int]]], 
) -> pd.DataFrame:
    out = frame.copy()

    for group, meta in GROUP_META.items():
        grid_id = group_grids[group]
        ws_col = f"ws117_ldaps_grid_{grid_id}"
        wd_col = f"wd117_ldaps_grid_{grid_id}"

        if ws_col not in out.columns or wd_col not in out.columns:
            raise KeyError(f"그룹 {group} 대표 grid={grid_id} 피처가 없습니다: {ws_col}, {wd_col}")

        season_lookups, global_lookup = shear_season_lookups[group]
        group_clusters = seasonal_clusters.get(f"kpx_group_{group}", {})

        if group_clusters:
            calibrated_ws = apply_shear_calibration_seasonal(
                out[ws_col], out["forecast_kst_dtm"], season_lookups, global_lookup, group_clusters,
            )
        else:
            calibrated_ws = apply_shear_calibration_lookup(out[ws_col], global_lookup)

        out[f"ws117_calibrated_group{group}"] = calibrated_ws

        # ==========================================
        # [신규 추가] 후류 효과(Wake Effect) 및 풍향-풍속 교호작용 피처
        # ==========================================
        wd_rad = np.radians(out[wd_col])
        out[f"wake_proxy_u_group{group}"] = calibrated_ws * np.cos(wd_rad)
        out[f"wake_proxy_v_group{group}"] = calibrated_ws * np.sin(wd_rad)
        out[f"wake_energy_u_group{group}"] = (calibrated_ws ** 3) * np.cos(wd_rad)
        out[f"wake_energy_v_group{group}"] = (calibrated_ws ** 3) * np.sin(wd_rad)

        curve = apply_group_hourly_power_quantile_lookup(calibrated_ws, power_lookups[group])
        base_name = meta["power_feature"]
        
        # 기존 모델/GAM과의 호환성을 위해 q50을 기본 이름으로 매핑
        out[base_name] = curve["q50"]
        out[f"{base_name}_q65"] = curve["q65"]
        out[f"{base_name}_q80"] = curve["q80"]
        out[f"{base_name}_trimmed_mean"] = curve["trimmed_mean"]
        out[f"{base_name}_mean"] = curve["mean"]

        out[meta["variability_feature"]] = apply_variability_lookup(
            calibrated_ws,
            out[wd_col],
            variability_lookups[group],
        )

    out["vestas_power_curve_pred"] = (
        out["vestas_power_curve_pred_group1"]
        + out["vestas_power_curve_pred_group2"]
    )
    return out


# -----------------------------------------------------------------------------
# 최종 데이터 생성
# -----------------------------------------------------------------------------
def _build_audit(
    mode: str,
    cutoff: pd.Timestamp,
    validation_start: pd.Timestamp | None,
    train_labels_all: pd.DataFrame,
    labels_used: pd.DataFrame,
    scada_vestas_all: pd.DataFrame,
    scada_vestas_fit: pd.DataFrame,
    scada_unison_all: pd.DataFrame,
    scada_unison_fit: pd.DataFrame,
) -> dict[str, object]:
    return {
        "mode": mode,
        "cutoff": str(cutoff),
        "validation_start": None if validation_start is None else str(validation_start),
        "label_rows_total": int(len(train_labels_all)),
        "label_rows_returned": int(len(labels_used)),
        "latest_label_returned": str(labels_used["kst_dtm"].max()),
        "vestas_rows_total": int(len(scada_vestas_all)),
        "vestas_rows_fit": int(len(scada_vestas_fit)),
        "latest_vestas_used": str(scada_vestas_fit["kst_dtm"].max()),
        "unison_rows_total": int(len(scada_unison_all)),
        "unison_rows_fit": int(len(scada_unison_fit)),
        "latest_unison_used": str(scada_unison_fit["kst_dtm"].max()),
    }


def get_tabular_data(
    mode: str = "final",
    validation_start: str | pd.Timestamp | None = None,
    prediction_reference_offset_hours: int = PREDICTION_REFERENCE_OFFSET_HOURS,
):
    """
    누수 방지형 학습·테스트 Tabular 데이터 생성.

    Parameters
    ----------
    mode:
        "final": 실제 2025 TEST 제출용. 첫 TEST 예보 기준시점 이전의
                 라벨과 SCADA만 사용한다.
        "validation": 시간 검증용. 첫 검증 예보 기준시점 이전 SCADA로만
                      룩업을 fit하고 전체 TRAIN을 반환한다.
    validation_start:
        mode="validation"일 때 첫 검증 대상 시각.
        실제 관측자료 cutoff는 이 target의 data_available_kst_dtm에서 계산한다.
    prediction_reference_offset_hours:
        기본 0. 대회가 공식적으로 기준시점을 data_available+1h로 확정했을 때만 1.

    Returns
    -------
    train_df, X_train, test_df, X_test, sample_sub

    검증 모드 사용법
    ---------------
    fit_mask = train_df["_fit_eligible"] & train_df[target].notna()
    valid_mask = train_df["_is_validation"] & train_df[target].notna()
    모델은 X_train.loc[fit_mask]로만 fit하고 X_train.loc[valid_mask]로 평가한다.
    """
    train_labels_all = pd.read_csv(
        RAW_DIR / "train" / "train_labels.csv",
        encoding="utf-8-sig",
    )
    sample_sub = pd.read_csv(
        RAW_DIR / "sample_submission.csv",
        encoding="utf-8-sig",
    )
    ldaps_train = pd.read_csv(
        RAW_DIR / "train" / "ldaps_train.csv",
        encoding="utf-8-sig",
    )
    gfs_train = pd.read_csv(
        RAW_DIR / "train" / "gfs_train.csv",
        encoding="utf-8-sig",
    )
    ldaps_test = pd.read_csv(
        RAW_DIR / "test" / "ldaps_test.csv",
        encoding="utf-8-sig",
    )
    gfs_test = pd.read_csv(
        RAW_DIR / "test" / "gfs_test.csv",
        encoding="utf-8-sig",
    )
    scada_vestas_all = pd.read_csv(
        RAW_DIR / "train" / "scada_vestas_train.csv",
        encoding="utf-8-sig",
    )
    scada_unison_all = pd.read_csv(
        RAW_DIR / "train" / "scada_unison_train.csv",
        encoding="utf-8-sig",
    )

    _parse_times(
        train_labels_all,
        sample_sub,
        ldaps_train,
        gfs_train,
        ldaps_test,
        gfs_test,
        scada_vestas_all,
        scada_unison_all,
    )

    for frame, name in (
        (ldaps_train, "LDAPS TRAIN"),
        (ldaps_test, "LDAPS TEST"),
        (gfs_train, "GFS TRAIN"),
        (gfs_test, "GFS TEST"),
    ):
        _assert_one_issue_per_target(frame, name)

    cutoff, actual_validation_start = _resolve_cutoff(
        mode=mode,
        validation_start=validation_start,
        ldaps_train=ldaps_train,
        gfs_train=gfs_train,
        ldaps_test=ldaps_test,
        gfs_test=gfs_test,
        prediction_reference_offset_hours=prediction_reference_offset_hours,
    )

    # 실제 발전량/운영자료로 만드는 모든 통계는 cutoff 이전만 허용한다.
    scada_vestas_fit = _filter_observed_before(scada_vestas_all, cutoff)
    scada_unison_fit = _filter_observed_before(scada_unison_all, cutoff)

    if mode == "final":
        labels_used = _filter_observed_before(train_labels_all, cutoff)
    else:
        labels_used = train_labels_all.copy()

    # TRAIN/TEST는 따로 결측 처리한다. 동일 예보 발행 묶음 안에서만 보간한다.
    ldaps_train = impute_ldaps_within_issue_cycle(ldaps_train)
    ldaps_test = impute_ldaps_within_issue_cycle(ldaps_test)

    ldaps_train_grid = add_ldaps_grid_features(ldaps_train)
    ldaps_test_grid = add_ldaps_grid_features(ldaps_test)
    ldaps_train_hourly, ldaps_test_hourly = build_ldaps_hourly_pair(
        ldaps_train_grid,
        ldaps_test_grid,
    )

    gfs_train_hourly = aggregate_weather(
        add_gfs_features(gfs_train),
        "gfs",
    )
    gfs_test_hourly = aggregate_weather(
        add_gfs_features(gfs_test),
        "gfs",
    )

    train_weather = ldaps_train_hourly.merge(
        gfs_train_hourly,
        on="forecast_kst_dtm",
        how="inner",
        validate="one_to_one",
    )
    test_weather = ldaps_test_hourly.merge(
        gfs_test_hourly,
        on="forecast_kst_dtm",
        how="inner",
        validate="one_to_one",
    )

    if len(train_weather) != train_labels_all["kst_dtm"].nunique():
        raise ValueError(
            "TRAIN 날씨 병합 행 수와 전체 라벨 시간 수가 다릅니다: "
            f"{len(train_weather)} != {train_labels_all['kst_dtm'].nunique()}"
        )
    if len(test_weather) != sample_sub["forecast_kst_dtm"].nunique():
        raise ValueError(
            "TEST 날씨 병합 행 수와 제출 시간 수가 다릅니다: "
            f"{len(test_weather)} != {sample_sub['forecast_kst_dtm'].nunique()}"
        )

    group_centers = get_kpx_group_centers(RAW_DIR / "info.xlsx")
    diagnose_grid_correlation(ldaps_train_grid, train_labels_all, cutoff)
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
    print(f"그룹별 대표 격자 선택({GRID_SELECTION_METHOD}): {group_grids}")

    try:  # [신규] regime 발견 때 쓰던 것과 동일 클러스터 재사용
        with open(MODEL_DIR / "seasonal_clusters.json", "r", encoding="utf-8") as f:
            seasonal_clusters = json.load(f)
    except FileNotFoundError:
        seasonal_clusters = {}
        print("[경고] seasonal_clusters.json 없음 -> shear 보정은 그룹 전체(글로벌) 기준으로만 동작합니다.")

    power_lookups: dict[int, pd.DataFrame] = {}
    variability_lookups: dict[int, pd.DataFrame] = {}
    shear_season_lookups: dict[int, tuple] = {}  # [신규]

    for group, meta in GROUP_META.items():
        scada_source = (
            scada_vestas_fit
            if meta["maker"] == "vestas"
            else scada_unison_fit
        )
        turbines = _turbine_names(meta["maker"], meta["turbines"])
        power_lookups[group] = fit_group_hourly_power_quantile_lookup(
            group=group,
            meta=meta,
            ldaps_train_grid=ldaps_train_grid,
            grid_id=group_grids[group],
            scada_fit=scada_source,
            cutoff=cutoff,
            ws_bin_width=0.5,
            min_count=30,
            trim_ratio=0.10
        )
        variability_lookups[group] = fit_variability_lookup(
            scada_source,
            turbines,
        )

        group_clusters = seasonal_clusters.get(f"kpx_group_{group}", {})
        if group_clusters:  # [신규] 계절별 학습, 없으면 기존 방식(글로벌만)
            shear_season_lookups[group] = fit_group_shear_calibration_seasonal(
                group, meta, ldaps_train_grid, group_grids[group],
                scada_vestas_fit, scada_unison_fit, cutoff, group_clusters,
            )
        else:
            gl = fit_group_shear_calibration(
                group, meta, ldaps_train_grid, group_grids[group],
                scada_vestas_fit, scada_unison_fit, cutoff,
            )
            shear_season_lookups[group] = ({}, gl)

    train_base = labels_used.rename(columns={"kst_dtm": "forecast_kst_dtm"})
    train_df = train_base.merge(
        train_weather,
        on="forecast_kst_dtm",
        how="left",
        validate="one_to_one",
    )
    test_df = sample_sub[["forecast_id", "forecast_kst_dtm"]].merge(
        test_weather,
        on="forecast_kst_dtm",
        how="left",
        validate="one_to_one",
    )

    train_df = _apply_scada_proxy_features(
        train_df,
        group_grids,
        power_lookups,
        variability_lookups,
        shear_season_lookups,
        seasonal_clusters,
    )
    test_df = _apply_scada_proxy_features(
        test_df,
        group_grids,
        power_lookups,
        variability_lookups,
        shear_season_lookups,
        seasonal_clusters,
    )

    # 검증 때 반드시 사용할 명시적 마스크. purge gap은 두 마스크 모두 False다.
    train_df["_fit_eligible"] = train_df["forecast_kst_dtm"] < cutoff
    if mode == "validation":
        train_df["_is_validation"] = (
            train_df["forecast_kst_dtm"] >= actual_validation_start
        )
    else:
        train_df["_is_validation"] = False

    non_feature_cols = {
        "forecast_kst_dtm",
        "ldaps_data_available_kst_dtm",
        "gfs_data_available_kst_dtm",
        "_fit_eligible",
        "_is_validation",
        *TARGET_COLS,
    }
    train_numeric = train_df.drop(
        columns=[col for col in non_feature_cols if col in train_df.columns]
    )
    train_numeric = train_numeric.select_dtypes(include=[np.number, "bool"])

    X_train = pd.concat(
        [calendar_features(train_df["forecast_kst_dtm"]), train_numeric],
        axis=1,
    )

    test_non_feature_cols = {
        "forecast_id",
        "forecast_kst_dtm",
        "ldaps_data_available_kst_dtm",
        "gfs_data_available_kst_dtm",
    }
    test_numeric = test_df.drop(
        columns=[col for col in test_non_feature_cols if col in test_df.columns]
    )
    test_numeric = test_numeric.select_dtypes(include=[np.number, "bool"])

    X_test = pd.concat(
        [calendar_features(test_df["forecast_kst_dtm"]), test_numeric],
        axis=1,
    )

    drop_cols = [c for c in EXCLUDED_FEATURES if c in X_train.columns]
    if drop_cols:
        print(f"제외 feature 적용: {drop_cols}")
        X_train = X_train.drop(columns=drop_cols)
        X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    missing_test_cols = [col for col in X_train.columns if col not in X_test.columns]
    extra_test_cols = [col for col in X_test.columns if col not in X_train.columns]
    if missing_test_cols or extra_test_cols:
        raise ValueError(
            "TRAIN/TEST 피처 컬럼이 다릅니다. "
            f"TEST 누락={missing_test_cols[:10]}, TEST 추가={extra_test_cols[:10]}"
        )
    X_test = X_test[X_train.columns]

    audit = _build_audit(
        mode=mode,
        cutoff=cutoff,
        validation_start=actual_validation_start,
        train_labels_all=train_labels_all,
        labels_used=labels_used,
        scada_vestas_all=scada_vestas_all,
        scada_vestas_fit=scada_vestas_fit,
        scada_unison_all=scada_unison_all,
        scada_unison_fit=scada_unison_fit,
    )
    train_df.attrs["leakage_audit"] = audit
    test_df.attrs["leakage_audit"] = audit

    return train_df, X_train, test_df, X_test, sample_sub


def get_target_xy(
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    target: str,
    subset: str = "fit",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    타깃 결측과 시간 검증 마스크를 함께 적용한다.

    subset="fit"       : 모델 학습 가능 구간
    subset="validation": 검증 구간
    subset="all"       : 타깃이 있는 전체 구간
    """
    if target not in TARGET_COLS:
        raise ValueError(f"알 수 없는 target입니다: {target}")

    mask = train_df[target].notna()
    if subset == "fit":
        mask &= train_df["_fit_eligible"]
    elif subset == "validation":
        mask &= train_df["_is_validation"]
    elif subset != "all":
        raise ValueError("subset은 'fit', 'validation', 'all' 중 하나여야 합니다.")

    return (
        X_train.loc[mask].reset_index(drop=True),
        train_df.loc[mask, target].reset_index(drop=True),
    )


# 검증 윈도우를 계절 단위로 자름
def get_bounded_validation_xy(train_df, X_train, target, validation_start, window_days=90):
    """validation_start부터 window_days일 동안만 잘라낸 '한 계절짜리' 검증 구간을 반환한다."""
    start_ts = pd.Timestamp(validation_start)
    end_ts = start_ts + pd.Timedelta(days=window_days)
    mask = (
        train_df[target].notna()
        & train_df["_is_validation"]
        & (train_df["forecast_kst_dtm"] >= start_ts)
        & (train_df["forecast_kst_dtm"] < end_ts)
    )
    return (
        X_train.loc[mask].reset_index(drop=True),
        train_df.loc[mask, target].reset_index(drop=True),
        train_df.loc[mask, "forecast_kst_dtm"].reset_index(drop=True),
    )


# Trimmed Mean 및 Quantile Power Curve 함수
def _trimmed_mean(values, trim_ratio: float = 0.10) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0: return np.nan
    arr = np.sort(arr)
    k = int(np.floor(len(arr) * trim_ratio))
    if k == 0 or len(arr) - 2 * k <= 0: return float(np.mean(arr))
    return float(np.mean(arr[k:-k]))


def fit_group_hourly_power_quantile_lookup(
    group: int, meta: dict, ldaps_train_grid: pd.DataFrame, grid_id: int,
    scada_fit: pd.DataFrame, cutoff: pd.Timestamp, ws_bin_width: float = 0.5,
    min_count: int = 30, trim_ratio: float = 0.10
) -> pd.DataFrame:
    turbines = _turbine_names(meta["maker"], meta["turbines"])
    power_cols = [f"{turbine}_power_kw10m" for turbine in turbines]
    
    scada = scada_fit[scada_fit["kst_dtm"] < cutoff][["kst_dtm"] + power_cols].copy()
    
    # 10분 단위 그룹 전체 발전량 합산
    scada["group_power_10m"] = scada[power_cols].sum(axis=1, min_count=len(power_cols))
    scada["forecast_kst_dtm"] = scada["kst_dtm"].dt.floor("h")
    
    # 1시간 단위 집계 (6개의 10분 데이터가 모두 있는 경우만)
    scada_hourly = scada.groupby("forecast_kst_dtm", as_index=False).agg(
        group_power_kwh=("group_power_10m", "sum"),
        n_10min=("group_power_10m", "count")
    )
    scada_hourly = scada_hourly[scada_hourly["n_10min"] == 6].copy()
    scada_hourly["group_power_kwh"] = scada_hourly["group_power_kwh"].clip(lower=0)
    
    ldaps_group = ldaps_train_grid[
        (ldaps_train_grid["grid_id"] == grid_id) & (ldaps_train_grid["forecast_kst_dtm"] < cutoff)
    ][["forecast_kst_dtm", "ws117_power_ldaps"]]
    
    merged = ldaps_group.merge(scada_hourly[["forecast_kst_dtm", "group_power_kwh"]], on="forecast_kst_dtm", how="inner").dropna()
    merged["ws_bin"] = np.floor(merged["ws117_power_ldaps"] / ws_bin_width)
    
    lookup = merged.groupby("ws_bin")["group_power_kwh"].agg(
        q50=lambda x: x.quantile(0.50),
        q65=lambda x: x.quantile(0.65),
        q80=lambda x: x.quantile(0.80),
        trimmed_mean=lambda x: _trimmed_mean(x, trim_ratio=trim_ratio),
        mean="mean", count="count"
    ).reset_index().sort_values("ws_bin")
    
    stat_cols = ["q50", "q65", "q80", "trimmed_mean", "mean"]
    lookup.loc[lookup["count"] < min_count, stat_cols] = np.nan
    lookup[stat_cols] = lookup[stat_cols].interpolate(limit_direction="both")
    lookup["ws_center"] = (lookup["ws_bin"] + 0.5) * ws_bin_width
    lookup = lookup.dropna(subset=["ws_center", "q50", "q65", "q80", "trimmed_mean"])
    
    # Quantile 역전 방지
    q_values = np.sort(lookup[["q50", "q65", "q80"]].to_numpy(), axis=1)
    lookup[["q50", "q65", "q80"]] = q_values
    
    return lookup[["ws_center", "q50", "q65", "q80", "trimmed_mean", "mean", "count"]]


def apply_group_hourly_power_quantile_lookup(wind_speed, lookup: pd.DataFrame) -> dict:
    ws = np.asarray(wind_speed, dtype=float)
    result = {}
    valid = np.isfinite(ws)
    x = lookup["ws_center"].to_numpy(dtype=float)
    
    for col in ["q50", "q65", "q80", "trimmed_mean", "mean"]:
        values = np.full(len(ws), np.nan, dtype=float)
        values[valid] = np.interp(ws[valid], x, lookup[col].to_numpy(dtype=float))
        result[col] = values
    return result