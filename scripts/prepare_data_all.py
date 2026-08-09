import pandas as pd
import re
import numpy as np
from config import RAW_DIR, TARGET_COLS

def aggregate_weather(df, prefix):
    """격자별 기상 데이터를 시간대별 평균으로 집계"""
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    drop_cols = {"data_available_kst_dtm", "grid_id", "latitude", "longitude"}
    value_cols = [c for c in df.columns if c not in {"forecast_kst_dtm", *drop_cols}]
    
    agg = df.groupby("forecast_kst_dtm")[value_cols].mean()
    agg.columns = [f"{prefix}_{c}_mean" for c in agg.columns]
    return agg.reset_index()

def calendar_features(dt_series):
    """기본 캘린더 피처 생성"""
    dt = pd.to_datetime(dt_series)
    out = pd.DataFrame(index=dt.index)
    out["month"] = dt.dt.month
    out["hour"] = dt.dt.hour
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out

########## 좌표/룩업 유틸 ##########
# DMS 좌표 문자열("37°16'55.61"N 128°57'02.10"E")을 십진수(위도, 경도)로 변환
def dms_to_decimal(dms_str):
    parts = dms_str.strip().split()
    def parse_one(token):
        m = re.match(r'''(\d+)°(\d+)'([\d.]+)"([NSEW])''', token)
        deg, minute, sec, direction = m.groups()
        value = float(deg) + float(minute) / 60 + float(sec) / 3600
        if direction in ("S", "W"):
            value = -value
        return value
    return parse_one(parts[0]), parse_one(parts[1])

# info.xlsx에서 KPX 그룹(1/2/3)별 터빈 중심 좌표 계산
def get_kpx_group_centers(info_xlsx_path):
    info = pd.read_excel(info_xlsx_path, sheet_name="info", header=3)
    info = info.loc[:, ~info.columns.astype(str).str.startswith("Unnamed")]
    info = info[info["호기"].notna()].reset_index(drop=True)
    info["KPX그룹"] = info["KPX그룹"].ffill()  # 병합 셀 -> 그룹 첫 터빈에만 값이 있어서 forward fill
    coords = info["좌표(Google)"].apply(dms_to_decimal)
    info["lat"] = coords.apply(lambda x: x[0])
    info["lon"] = coords.apply(lambda x: x[1])
    return info.groupby("KPX그룹")[["lat", "lon"]].mean()

# 중심 좌표와 가장 가까운 LDAPS 격자(grid_id) 탐색
def nearest_ldaps_grid_id(center_lat, center_lon, ldaps_df):
    grids = ldaps_df[["grid_id", "latitude", "longitude"]].drop_duplicates()
    dist = np.sqrt((grids["latitude"] - center_lat) ** 2 + (grids["longitude"] - center_lon) ** 2)
    return int(grids.loc[dist.idxmin(), "grid_id"])

# 중심 좌표와 가장 가까운 GFS 격자(grid_id) 탐색
def nearest_gfs_grid_id(center_lat, center_lon, gfs_df):
    grids = gfs_df[["grid_id", "latitude", "longitude"]].drop_duplicates()
    dist = np.sqrt((grids["latitude"] - center_lat) ** 2 + (grids["longitude"] - center_lon) ** 2)
    return int(grids.loc[dist.idxmin(), "grid_id"])

# 풍속bin -> 터빈 1기당 중앙값 발전량 룩업 테이블 생성 (train SCADA 전용, fit 단계)
# 주의: 그룹 내 터빈들의 (발전량, 풍속) 쌍을 전부 풀링해서 "터빈 1기 기준" 중앙값으로 만듦
#       (원래 명세서 로직은 터빈별 실측 풍속으로 각자 계산해 합산하지만,
#        apply 단계에서는 LDAPS 예보 풍속 1개만 쓸 수 있어서 이렇게 단순화함)
def fit_power_curve_lookup(scada_train, turbines, power_suffix="_power_kw10m", ws_suffix="_ws",
                            ws_bin_width=0.5, min_count=30):
    pooled = pd.concat([
        pd.DataFrame({
            "power": scada_train[f"{wtg}{power_suffix}"],
            "ws": scada_train[f"{wtg}{ws_suffix}"],
        })
        for wtg in turbines
    ], ignore_index=True).dropna()

    pooled["ws_bin"] = np.floor(pooled["ws"] / ws_bin_width)
    lookup = pooled.groupby("ws_bin")["power"].agg(["median", "count"]).reset_index()
    lookup.loc[lookup["count"] < min_count, "median"] = np.nan
    lookup["median"] = lookup["median"].interpolate(limit_direction="both")
    lookup["ws_center"] = (lookup["ws_bin"] + 0.5) * ws_bin_width
    return lookup[["ws_center", "median"]]

def fit_power_curve_lookup_gfs(scada_train, turbines, v_in, v_rated, v_out, capacity,
                                power_suffix="_power_kw10m", ws_suffix="_ws",
                                ws_bin_width=0.5, min_count=30):
    pooled = pd.concat([
        pd.DataFrame({
            "power": scada_train[f"{wtg}{power_suffix}"],
            "ws": scada_train[f"{wtg}{ws_suffix}"],
        })
        for wtg in turbines
    ], ignore_index=True).dropna()

    pooled["ws_bin"] = np.floor(pooled["ws"] / ws_bin_width)
    observed = pooled.groupby("ws_bin")["power"].agg(["median", "count"]).reset_index()

    # 관측이 아예 없는 고풍속(컷아웃 이후 포함) 구간까지 bin을 넉넉히 만들어둠
    max_bin = max(int(observed["ws_bin"].max()) if len(observed) else 0,
                  int(np.ceil(v_out / ws_bin_width)) + 4)
    lookup = pd.DataFrame({"ws_bin": np.arange(0, max_bin + 1)}).merge(observed, on="ws_bin", how="left")
    lookup["count"] = lookup["count"].fillna(0)
    lookup["ws_center"] = (lookup["ws_bin"] + 0.5) * ws_bin_width

    reliable = lookup["count"] >= min_count
    lookup.loc[~reliable, "median"] = np.nan

    # 이론적 파워커브 (헬퍼 함수 분리 없이 인라인 계산)
    ws = lookup["ws_center"].to_numpy(dtype=float)
    theoretical = np.zeros_like(ws)
    ramp = (ws >= v_in) & (ws < v_rated)
    theoretical[ramp] = capacity * ((ws[ramp] - v_in) / (v_rated - v_in)) ** 3
    flat = (ws >= v_rated) & (ws < v_out)
    theoretical[flat] = capacity

    # 정격구간(표본 충분한 bin)에서 실측/이론 비율(감쇄율)을 구해서 표본 부족 구간에 곱해서 대체
    rated_mask = reliable.to_numpy() & (ws >= v_rated) & (ws < v_out)
    if rated_mask.any():
        derate_ratio = (lookup.loc[rated_mask, "median"].to_numpy() / theoretical[rated_mask]).mean()
    else:
        derate_ratio = 1.0
    fallback = theoretical * derate_ratio
    lookup["median"] = lookup["median"].fillna(pd.Series(fallback, index=lookup.index))

    return lookup[["ws_center", "median"]]

# 룩업 테이블 + 예보 풍속(LDAPS 등)으로 그룹 전체 발전량 추정 (apply 단계, train/test 공통)
def apply_power_curve_lookup(wind_speed, lookup, n_turbines):
    per_turbine_pred = np.interp(wind_speed, lookup["ws_center"], lookup["median"])
    return per_turbine_pred * n_turbines

# 풍속bin×풍향bin -> 터빈 1기당 발전량 표준편차 룩업 테이블 생성 (train SCADA 전용, fit 단계)
# fit_power_curve_lookup과 동일하게 그룹 내 터빈들을 풀링해서 "터빈 1기 기준" 값으로 만듦
def fit_variability_lookup(scada_train, turbines, power_suffix="_power_kw10m", ws_suffix="_ws", wd_suffix="_wd",
                            ws_bin_width=0.5, wd_bin_width=30, min_count=30):
    pooled = pd.concat([
        pd.DataFrame({
            "power": scada_train[f"{wtg}{power_suffix}"],
            "ws": scada_train[f"{wtg}{ws_suffix}"],
            "wd": scada_train[f"{wtg}{wd_suffix}"] % 360,
        })
        for wtg in turbines
    ], ignore_index=True).dropna()

    pooled["ws_bin"] = np.floor(pooled["ws"] / ws_bin_width)
    pooled["wd_bin"] = np.floor(pooled["wd"] / wd_bin_width)
    lookup = pooled.groupby(["ws_bin", "wd_bin"])["power"].agg(["std", "count"]).reset_index()
    lookup.loc[lookup["count"] < min_count, "std"] = np.nan
    return lookup[["ws_bin", "wd_bin", "std"]]

# 룩업 테이블 + 예보 풍속·풍향(LDAPS)으로 그룹 전체 변동성 추정 (apply 단계, train/test 공통)
# 해당 (풍속bin, 풍향bin) 표본이 부족해 std가 NaN이면, 풍향은 무시하고 풍속bin 평균으로 대체(fallback)
def apply_variability_lookup(wind_speed, wind_direction, lookup, ws_bin_width=0.5, wd_bin_width=30):
    query = pd.DataFrame({
        "ws_bin": np.floor(np.asarray(wind_speed) / ws_bin_width),
        "wd_bin": np.floor((np.asarray(wind_direction) % 360) / wd_bin_width),
    })
    merged = query.merge(lookup, on=["ws_bin", "wd_bin"], how="left")
    fallback = lookup.groupby("ws_bin")["std"].mean().rename("std_fallback")
    merged = merged.merge(fallback, on="ws_bin", how="left")
    return merged["std"].fillna(merged["std_fallback"]).values


########## tabular 데이터 생성 ##########
def get_tabular_data():
    """학습 및 테스트용 Tabular 데이터셋 병합 및 반환"""
    # 1. 데이터 로드
    train_labels = pd.read_csv(RAW_DIR / "train" / "train_labels.csv", encoding="utf-8-sig")
    sample_sub = pd.read_csv(RAW_DIR / "sample_submission.csv", encoding="utf-8-sig")
    
    ldaps_train = pd.read_csv(RAW_DIR / "train" / "ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv(RAW_DIR / "train" / "gfs_train.csv", encoding="utf-8-sig")
    ldaps_test = pd.read_csv(RAW_DIR / "test" / "ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv(RAW_DIR / "test" / "gfs_test.csv", encoding="utf-8-sig")
    scada_vestas_train = pd.read_csv(RAW_DIR / "train" / "scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison_train = pd.read_csv(RAW_DIR / "train" / "scada_unison_train.csv", encoding="utf-8-sig")

    train_labels["kst_dtm"] = pd.to_datetime(train_labels["kst_dtm"])
    sample_sub["forecast_kst_dtm"] = pd.to_datetime(sample_sub["forecast_kst_dtm"])
    scada_vestas_train["kst_dtm"] = pd.to_datetime(scada_vestas_train["kst_dtm"])
    scada_unison_train["kst_dtm"] = pd.to_datetime(scada_unison_train["kst_dtm"])

    ldaps_train["forecast_kst_dtm"] = pd.to_datetime(ldaps_train["forecast_kst_dtm"])   # ← 추가
    ldaps_test["forecast_kst_dtm"] = pd.to_datetime(ldaps_test["forecast_kst_dtm"]) 
    gfs_train["forecast_kst_dtm"] = pd.to_datetime(gfs_train["forecast_kst_dtm"])
    gfs_test["forecast_kst_dtm"] = pd.to_datetime(gfs_test["forecast_kst_dtm"])

    # 2. feature engineering 적용
    ###### GFS ######
    gfs_train_grid = gfs_train.pipe(add_ws117_power_gfs)
    gfs_test_grid = gfs_test.pipe(add_ws117_power_gfs)

    # GFS Train
    gfs_train = (
        gfs_train
        # .pipe(add_wd_cos)
        # .pipe(add_wd_sin)
        .pipe(add_turbulence_intensity)
        .pipe(add_dayofyear_cos)
        .pipe(add_dayofyear_sin)
        # .pipe(add_hour_cos)
        # .pipe(add_hour_sin)
        # .pipe(add_ws117_channel_cross)
        # .pipe(add_ws117_channel_along)
    )
    gfs_train_ws117 = (
            add_ws117_gfs_grid(gfs_train_grid)
            .pipe(add_ws117_gfs_spatial_mean)
            .pipe(add_ws117_cube_gfs)
            .pipe(add_ws117_diff_1h_gfs)
            .pipe(add_ws117_lag_24h_gfs)
            .pipe(add_ws117_roll_mean_6h_gfs)
            .pipe(add_ws117_roll_std_6h_gfs)
    )
    

    # GFS Test
    gfs_test = (
        gfs_test
        # .pipe(add_wd_cos)
        # .pipe(add_wd_sin)
        .pipe(add_turbulence_intensity)
        .pipe(add_dayofyear_cos)
        .pipe(add_dayofyear_sin)
        # .pipe(add_hour_cos)
        # .pipe(add_hour_sin)
        # .pipe(add_ws117_channel_cross)
        # .pipe(add_ws117_channel_along)
    )
    gfs_test_ws117 = (
        add_ws117_gfs_grid(gfs_test_grid)
        .pipe(add_ws117_gfs_spatial_mean)
        .pipe(add_ws117_cube_gfs)
        .pipe(add_ws117_diff_1h_gfs)
        .pipe(add_ws117_lag_24h_gfs)
        .pipe(add_ws117_roll_mean_6h_gfs)
        .pipe(add_ws117_roll_std_6h_gfs)
    )

    gfs_train_alpha = add_alpha_shear_gfs(gfs_train)
    gfs_train_density = add_air_density_gfs(gfs_train)
    gfs_train_prmsl = add_prmsl_range_gfs(gfs_train)
    gfs_test_alpha = add_alpha_shear_gfs(gfs_test)
    gfs_test_density = add_air_density_gfs(gfs_test)
    gfs_test_prmsl = add_prmsl_range_gfs(gfs_test)

    gfs_train_hourly = (
        gfs_train_ws117
        .merge(gfs_train_alpha, on="forecast_kst_dtm", how="left")
        .merge(gfs_train_density, on="forecast_kst_dtm", how="left")
        .merge(gfs_train_prmsl, on="forecast_kst_dtm", how="left")
        .pipe(add_wind_energy_flux_gfs)
    )
    gfs_test_hourly = (
        gfs_test_ws117
        .merge(gfs_test_alpha, on="forecast_kst_dtm", how="left")
        .merge(gfs_test_density, on="forecast_kst_dtm", how="left")
        .merge(gfs_test_prmsl, on="forecast_kst_dtm", how="left")
        .pipe(add_wind_energy_flux_gfs)
    )

    ###### LDAPS ######
    ldaps_train_grid = ldaps_train.pipe(add_ws117_power_ldaps).pipe(add_wd117_ldaps)
    ldaps_test_grid = ldaps_test.pipe(add_ws117_power_ldaps).pipe(add_wd117_ldaps)

    # LDAPS Train
    ldaps_train_ws117 = (
        add_ws117_ldaps_grid(ldaps_train_grid)
        .merge(add_wd117_ldaps_grid(ldaps_train_grid), on="forecast_kst_dtm", how="left")
        .pipe(add_ws117_ldaps_spatial_mean)
        .pipe(add_ws117_cube_ldaps)
        .pipe(add_ws117_diff_1h_ldaps)
        .pipe(add_ws117_lag_24h_ldaps)
        .pipe(add_ws117_roll_mean_6h_ldaps)
        .pipe(add_ws117_roll_std_6h_ldaps)
    )

    ldaps_test_ws117 = (
        add_ws117_ldaps_grid(ldaps_test_grid)
        .merge(add_wd117_ldaps_grid(ldaps_test_grid), on="forecast_kst_dtm", how="left")
        .pipe(add_ws117_ldaps_spatial_mean)
        .pipe(add_ws117_cube_ldaps)
        .pipe(add_ws117_diff_1h_ldaps)
        .pipe(add_ws117_lag_24h_ldaps)
        .pipe(add_ws117_roll_mean_6h_ldaps)
        .pipe(add_ws117_roll_std_6h_ldaps)
    )

    # pipe 불가능 항목들
    ldaps_train_alpha = add_alpha_shear_ldaps(ldaps_train)
    ldaps_train_density = add_air_density_ldaps(ldaps_train)
    ldaps_train_prmsl = add_prmsl_range_ldaps(ldaps_train)
    ldaps_test_alpha = add_alpha_shear_ldaps(ldaps_test)
    ldaps_test_density = add_air_density_ldaps(ldaps_test)
    ldaps_test_prmsl = add_prmsl_range_ldaps(ldaps_test)

    ldaps_train_hourly = (
        ldaps_train_ws117
        .merge(ldaps_train_alpha, on="forecast_kst_dtm", how="left")
        .merge(ldaps_train_density, on="forecast_kst_dtm", how="left")
        .merge(ldaps_train_prmsl, on="forecast_kst_dtm", how="left")
        .pipe(add_wind_energy_flux_ldaps)
    )
    ldaps_test_hourly = (
        ldaps_test_ws117
        .merge(ldaps_test_alpha, on="forecast_kst_dtm", how="left")
        .merge(ldaps_test_density, on="forecast_kst_dtm", how="left")
        .merge(ldaps_test_prmsl, on="forecast_kst_dtm", how="left")
        .pipe(add_wind_energy_flux_ldaps)
    )

    ###### SCADA -> 그룹별 발전량 룩업 (fit/apply) ######
    # TODO: 지금은 그룹 내 터빈 좌표를 "단순 평균"해서 중심좌표로 쓰고 있음.
    #       나중엔 이 그룹의 발전량에 가장 영향을 주는 좌표로 개선하면 좋겠음.
    group_centers = get_kpx_group_centers(RAW_DIR / "info.xlsx")
    grid_g1 = nearest_ldaps_grid_id(*group_centers.loc[1], ldaps_train)  # VESTAS group1
    grid_g2 = nearest_ldaps_grid_id(*group_centers.loc[2], ldaps_train)  # VESTAS group2
    grid_g3 = nearest_ldaps_grid_id(*group_centers.loc[3], ldaps_train)  # UNISON group3

    gfs_g1 = nearest_gfs_grid_id(*group_centers.loc[1], gfs_train)  # VESTAS group1
    gfs_g2 = nearest_gfs_grid_id(*group_centers.loc[2], gfs_train)  # VESTAS group2
    gfs_g3 = nearest_gfs_grid_id(*group_centers.loc[3], gfs_train)  # UNISON group3

    # fit: train SCADA로 그룹별 풍속->발전량 룩업 학습
    lookup_vestas_all = fit_power_curve_lookup(scada_vestas_train, [f"vestas_wtg{i:02d}" for i in range(1, 13)])
    lookup_vestas_g1 = fit_power_curve_lookup(scada_vestas_train, [f"vestas_wtg{i:02d}" for i in range(1, 7)])
    lookup_vestas_g2 = fit_power_curve_lookup(scada_vestas_train, [f"vestas_wtg{i:02d}" for i in range(7, 13)])
    lookup_unison = fit_power_curve_lookup(scada_unison_train, [f"unison_wtg{i:02d}" for i in range(1, 6)])

    # 발전량 변동성에 fit/apply 방식 적용
    lookup_var_vestas_g1 = fit_variability_lookup(scada_vestas_train, [f"vestas_wtg{i:02d}" for i in range(1, 7)])
    lookup_var_vestas_g2 = fit_variability_lookup(scada_vestas_train, [f"vestas_wtg{i:02d}" for i in range(7, 13)])
    lookup_var_unison = fit_variability_lookup(scada_unison_train, [f"unison_wtg{i:02d}" for i in range(1, 6)])

    lookup_gfs_vestas_g1 = fit_power_curve_lookup_gfs(
        scada_vestas_train, [f"vestas_wtg{i:02d}" for i in range(1, 7)],
        v_in=3, v_rated=12.0, v_out=22.5, capacity=21600
    )
    lookup_gfs_vestas_g2 = fit_power_curve_lookup_gfs(
        scada_vestas_train, [f"vestas_wtg{i:02d}" for i in range(7, 13)],
        v_in=3, v_rated=12.0, v_out=22.5, capacity=21600
    )
    lookup_gfs_unison = fit_power_curve_lookup_gfs(
        scada_unison_train, [f"unison_wtg{i:02d}" for i in range(1, 6)],
        v_in=3, v_rated=12.5, v_out=22.0, capacity=21000
    )

    # 터빈 효율 (그룹당 상수 1개, train SCADA 전체로 학습 -> 모든 행에 동일하게 broadcast, 확인 완료)
    vestas_eff1 = add_vestas_turbine_efficiency_group1(scada_vestas_train)["vestas_turbine_efficiency_group1"].iloc[0]
    vestas_eff2 = add_vestas_turbine_efficiency_group2(scada_vestas_train)["vestas_turbine_efficiency_group2"].iloc[0]
    unison_eff = add_unison_turbine_efficiency(scada_unison_train)["unison_turbine_efficiency"].iloc[0]

    
    # 날씨 데이터 병합 (기존 로직 + LDAPS 파생변수 merge)
    train_weather = (
        aggregate_weather(ldaps_train, "ldaps")
        .merge(aggregate_weather(gfs_train, "gfs"), on="forecast_kst_dtm", how="inner")
        .merge(ldaps_train_hourly, on="forecast_kst_dtm", how="left")
        .merge(gfs_train_hourly, on="forecast_kst_dtm", how="left")
    )
    test_weather = (
        aggregate_weather(ldaps_test, "ldaps")
        .merge(aggregate_weather(gfs_test, "gfs"), on="forecast_kst_dtm", how="inner")
        .merge(ldaps_test_hourly, on="forecast_kst_dtm", how="left")
        .merge(gfs_test_hourly, on="forecast_kst_dtm", how="left")
    )

    train_base = train_labels.rename(columns={"kst_dtm": "forecast_kst_dtm"})
    train_df = train_base.merge(train_weather, on="forecast_kst_dtm", how="left")
    train_df["vestas_turbine_efficiency_group1"] = vestas_eff1
    train_df["vestas_turbine_efficiency_group2"] = vestas_eff2
    train_df["unison_turbine_efficiency"] = unison_eff

    # apply: 룩업 + LDAPS 격자 풍속으로 train/test 둘 다 채움
    train_df["vestas_power_curve_pred"] = apply_power_curve_lookup(
        train_df[f"ws117_ldaps_grid_{grid_g1}"], lookup_vestas_all, n_turbines=12
    )
    train_df["vestas_power_curve_pred_group1"] = apply_power_curve_lookup(
        train_df[f"ws117_ldaps_grid_{grid_g1}"], lookup_vestas_g1, n_turbines=6
    )
    train_df["vestas_power_curve_pred_group2"] = apply_power_curve_lookup(
        train_df[f"ws117_ldaps_grid_{grid_g2}"], lookup_vestas_g2, n_turbines=6
    )
    train_df["unison_power_curve_pred"] = apply_power_curve_lookup(
        train_df[f"ws117_ldaps_grid_{grid_g3}"], lookup_unison, n_turbines=5
    )
    train_df["vestas_power_variability_group1"] = apply_variability_lookup(
        train_df[f"ws117_ldaps_grid_{grid_g1}"], train_df[f"wd117_ldaps_grid_{grid_g1}"], lookup_var_vestas_g1
    )
    train_df["vestas_power_variability_group2"] = apply_variability_lookup(
        train_df[f"ws117_ldaps_grid_{grid_g2}"], train_df[f"wd117_ldaps_grid_{grid_g2}"], lookup_var_vestas_g2
    )
    train_df["unison_power_variability"] = apply_variability_lookup(
        train_df[f"ws117_ldaps_grid_{grid_g3}"], train_df[f"wd117_ldaps_grid_{grid_g3}"], lookup_var_unison
    )
    train_df = add_power_curve_pred_group1_ldaps(train_df, grid_g1)
    train_df = add_power_curve_pred_group2_ldaps(train_df, grid_g2)
    train_df = add_power_curve_pred_group3_ldaps(train_df, grid_g3)
    train_df = add_power_curve_pred_group1_gfs(train_df, gfs_g1)
    train_df = add_power_curve_pred_group2_gfs(train_df, gfs_g2)
    train_df = add_power_curve_pred_group3_gfs(train_df, gfs_g3)
    train_df["power_curve_pred_lookup_group1_gfs"] = apply_power_curve_lookup(
        train_df[f"ws117_gfs_grid_{gfs_g1}"], lookup_gfs_vestas_g1, n_turbines=6
    )
    train_df["power_curve_pred_lookup_group2_gfs"] = apply_power_curve_lookup(
        train_df[f"ws117_gfs_grid_{gfs_g2}"], lookup_gfs_vestas_g2, n_turbines=6
    )
    train_df["power_curve_pred_lookup_group3_gfs"] = apply_power_curve_lookup(
        train_df[f"ws117_gfs_grid_{gfs_g3}"], lookup_gfs_unison, n_turbines=5
    )

    test_df = sample_sub[["forecast_id", "forecast_kst_dtm"]].merge(
        test_weather, on="forecast_kst_dtm", how="left"
    )
    test_df["vestas_power_curve_pred"] = apply_power_curve_lookup(
        test_df[f"ws117_ldaps_grid_{grid_g1}"], lookup_vestas_all, n_turbines=12
    )
    test_df["vestas_power_curve_pred_group1"] = apply_power_curve_lookup(
        test_df[f"ws117_ldaps_grid_{grid_g1}"], lookup_vestas_g1, n_turbines=6
    )
    test_df["vestas_power_curve_pred_group2"] = apply_power_curve_lookup(
        test_df[f"ws117_ldaps_grid_{grid_g2}"], lookup_vestas_g2, n_turbines=6
    )
    test_df["unison_power_curve_pred"] = apply_power_curve_lookup(
        test_df[f"ws117_ldaps_grid_{grid_g3}"], lookup_unison, n_turbines=5
    )
    test_df["vestas_power_variability_group1"] = apply_variability_lookup(
        test_df[f"ws117_ldaps_grid_{grid_g1}"], test_df[f"wd117_ldaps_grid_{grid_g1}"], lookup_var_vestas_g1
    )
    test_df["vestas_power_variability_group2"] = apply_variability_lookup(
        test_df[f"ws117_ldaps_grid_{grid_g2}"], test_df[f"wd117_ldaps_grid_{grid_g2}"], lookup_var_vestas_g2
    )
    test_df["unison_power_variability"] = apply_variability_lookup(
        test_df[f"ws117_ldaps_grid_{grid_g3}"], test_df[f"wd117_ldaps_grid_{grid_g3}"], lookup_var_unison
    )
    test_df["vestas_turbine_efficiency_group1"] = vestas_eff1
    test_df["vestas_turbine_efficiency_group2"] = vestas_eff2
    test_df["unison_turbine_efficiency"] = unison_eff
    test_df = add_power_curve_pred_group1_ldaps(test_df, grid_g1)
    test_df = add_power_curve_pred_group2_ldaps(test_df, grid_g2)
    test_df = add_power_curve_pred_group3_ldaps(test_df, grid_g3)
    test_df = add_power_curve_pred_group1_gfs(test_df, gfs_g1)
    test_df = add_power_curve_pred_group2_gfs(test_df, gfs_g2)
    test_df = add_power_curve_pred_group3_gfs(test_df, gfs_g3)
    test_df["power_curve_pred_lookup_group1_gfs"] = apply_power_curve_lookup(
        test_df[f"ws117_gfs_grid_{gfs_g1}"], lookup_gfs_vestas_g1, n_turbines=6
    )
    test_df["power_curve_pred_lookup_group2_gfs"] = apply_power_curve_lookup(
        test_df[f"ws117_gfs_grid_{gfs_g2}"], lookup_gfs_vestas_g2, n_turbines=6
    )
    test_df["power_curve_pred_lookup_group3_gfs"] = apply_power_curve_lookup(
        test_df[f"ws117_gfs_grid_{gfs_g3}"], lookup_gfs_unison, n_turbines=5
    )

    X_train = pd.concat([
        calendar_features(train_df["forecast_kst_dtm"]),
        train_df.drop(columns=["forecast_kst_dtm", *TARGET_COLS])
    ], axis=1)

    X_test = pd.concat([
        calendar_features(test_df["forecast_kst_dtm"]),
        test_df.drop(columns=["forecast_id", "forecast_kst_dtm"])
    ], axis=1)
    X_test = X_test[X_train.columns]

    return train_df, X_train, test_df, X_test, sample_sub

########## LDAPS ##########
# LDAPS 풍속 시어 지수 (grid_id별 행 단위)
def add_alpha_shear_ldaps(df):
    df = df.copy()
    u10 = df["heightAboveGround_10_10u"]
    v10 = df["heightAboveGround_10_10v"]
    ws10 = np.sqrt(u10**2 + v10**2)

    u50 = (df["heightAboveGround_50_50MUmax"] + df["heightAboveGround_50_50MUmin"]) / 2
    v50 = (df["heightAboveGround_50_50MVmax"] + df["heightAboveGround_50_50MVmin"]) / 2
    ws50 = np.sqrt(u50**2 + v50**2)

    alpha = np.log((ws50 + 1e-8) / (ws10 + 1e-8)) / np.log(50 / 10)
    df["alpha_shear_ldaps"] = np.clip(alpha, -0.5, 1.0)
    out = df.groupby("forecast_kst_dtm", as_index=False)["alpha_shear_ldaps"].mean()
    return out

# 117m 허브 높이 보간 풍속 
# (grid_id별 행 단위, alpha_shear_ldaps 재계산 포함)
def add_ws117_power_ldaps(df):
    u10 = df["heightAboveGround_10_10u"]
    v10 = df["heightAboveGround_10_10v"]
    ws10 = np.sqrt(u10**2 + v10**2)

    u50 = (df["heightAboveGround_50_50MUmax"] + df["heightAboveGround_50_50MUmin"]) / 2
    v50 = (df["heightAboveGround_50_50MVmax"] + df["heightAboveGround_50_50MVmin"]) / 2
    ws50 = np.sqrt(u50**2 + v50**2)

    alpha = np.clip(np.log((ws50 + 1e-8) / (ws10 + 1e-8)) / np.log(50 / 10), -0.5, 1.0)
    scale = (117 / 50) ** alpha
    u117 = u50 * scale
    v117 = v50 * scale
    df["ws117_power_ldaps"] = np.sqrt(u117**2 + v117**2)
    return df

# LDAPS 117m 보정 풍향 (도 단위 0~360, grid_id별 행 단위)
# ws117_power_ldaps와 같은 u117/v117 계산을 다시 하지만(중복 허용 방침), 이번엔 크기 대신 방향을 저장
def add_wd117_ldaps(df):
    u10 = df["heightAboveGround_10_10u"]
    v10 = df["heightAboveGround_10_10v"]
    ws10 = np.sqrt(u10**2 + v10**2)

    u50 = (df["heightAboveGround_50_50MUmax"] + df["heightAboveGround_50_50MUmin"]) / 2
    v50 = (df["heightAboveGround_50_50MVmax"] + df["heightAboveGround_50_50MVmin"]) / 2
    ws50 = np.sqrt(u50**2 + v50**2)

    alpha = np.clip(np.log((ws50 + 1e-8) / (ws10 + 1e-8)) / np.log(50 / 10), -0.5, 1.0)
    scale = (117 / 50) ** alpha
    u117 = u50 * scale
    v117 = v50 * scale

    # 기상학적 관례(바람이 "불어오는" 방향) + 0~360도로 보정
    df["wd117_ldaps"] = np.degrees(np.arctan2(-u117, -v117)) % 360
    return df

# LDAPS 국소 격자별 117m 보정 풍향 pivot (i = 1~16), add_ws117_ldaps_grid와 동일한 방식
def add_wd117_ldaps_grid(df):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    grid_wide = df.pivot(index="forecast_kst_dtm", columns="grid_id", values="wd117_ldaps")
    grid_wide.columns = [f"wd117_ldaps_grid_{int(g)}" for g in grid_wide.columns]
    return grid_wide.reset_index()

# LDAPS 국소 공기 밀도 
# (grid_id별 행 단위)
def add_air_density_ldaps(df):
    df = df.copy()
    df["air_density_ldaps"] = df["surface_0_sp"] / (287.058 * df["heightAboveGround_2_t"])
    out = df.groupby("forecast_kst_dtm", as_index=False)["air_density_ldaps"].mean()
    return out

# LDAPS 전격자 117m 풍속 공간 평균 
# (시간당 1행으로 변환)
def add_ws117_ldaps_spatial_mean(df):
    grid_cols = [c for c in df.columns if c.startswith("ws117_ldaps_grid_")]
    df["ws117_ldaps_spatial_mean"] = df[grid_cols].mean(axis=1)
    return df

# LDAPS 국소 기압경도 
# (해면기압 격자간 차이, 시간당 1행으로 변환)
def add_prmsl_range_ldaps(df):
    out = df.groupby("forecast_kst_dtm")["meanSea_0_prmsl"].agg(lambda x: x.max() - x.min())
    out = out.rename("prmsl_range_ldaps").reset_index()
    return out

# 117m 풍속 3승 변수 
# (시간당 1행, ws117_ldaps_spatial_mean 필요)
def add_ws117_cube_ldaps(df):
    df["ws117_cube_ldaps"] = df["ws117_ldaps_spatial_mean"] ** 3
    return df

# 바람 운동 에너지 플럭스 
# (시간당 1행, air_density_ldaps・ws117_cube_ldaps 필요)
def add_wind_energy_flux_ldaps(df):
    df["wind_energy_flux_ldaps"] = 0.5 * df["air_density_ldaps"] * df["ws117_cube_ldaps"]
    return df

# 117m 풍속 1시간 차분 
# (시간당 1행, ws117_ldaps_spatial_mean 필요)
def add_ws117_diff_1h_ldaps(df):
    df = df.sort_values("forecast_kst_dtm").reset_index(drop=True)
    df["ws117_diff_1h_ldaps"] = df["ws117_ldaps_spatial_mean"].diff(1)
    return df

# 24시간 전 예보 풍속 
# (시간당 1행, ws117_ldaps_spatial_mean 필요)
def add_ws117_lag_24h_ldaps(df):
    df = df.sort_values("forecast_kst_dtm").reset_index(drop=True)
    df["ws117_lag_24h_ldaps"] = df["ws117_ldaps_spatial_mean"].shift(24)
    return df

# 6시간 롤링 이동평균 
# (시간당 1행, ws117_ldaps_spatial_mean 필요)
def add_ws117_roll_mean_6h_ldaps(df):
    df = df.sort_values("forecast_kst_dtm").reset_index(drop=True)
    df["ws117_roll_mean_6h_ldaps"] = df["ws117_ldaps_spatial_mean"].rolling(window=6, min_periods=1).mean()
    return df

# 6시간 롤링 표준편차 
# (시간당 1행, ws117_ldaps_spatial_mean 필요)
def add_ws117_roll_std_6h_ldaps(df):
    df = df.sort_values("forecast_kst_dtm").reset_index(drop=True)
    df["ws117_roll_std_6h_ldaps"] = df["ws117_ldaps_spatial_mean"].rolling(window=6, min_periods=2).std()
    return df

# LDAPS 국소 격자별 117m 보간 풍속 (i = 1~16)
def add_ws117_ldaps_grid(df):
    df = df.copy()  
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    grid_wide = df.pivot(index="forecast_kst_dtm", columns="grid_id", values="ws117_power_ldaps")
    grid_wide.columns = [f"ws117_ldaps_grid_{int(g)}" for g in grid_wide.columns]
    return grid_wide.reset_index()


########## GFS ##########
# GFS 풍속 시어 지수 (grid_id별 행 단위, 100m/10m 기준)
def add_alpha_shear_gfs(df):
    u10 = df["heightAboveGround_10_10u"]
    v10 = df["heightAboveGround_10_10v"]
    ws10 = np.sqrt(u10**2 + v10**2)

    u100 = df["heightAboveGround_100_100u"]
    v100 = df["heightAboveGround_100_100v"]
    ws100 = np.sqrt(u100**2 + v100**2)

    alpha = np.log((ws100 + 1e-8) / (ws10 + 1e-8)) / np.log(100 / 10)
    df["alpha_shear_gfs"] = np.clip(alpha, -0.5, 1.0)
    out = df.groupby("forecast_kst_dtm", as_index=False)["alpha_shear_gfs"].mean()
    return out

# 117m 허브 높이 보간 풍속 (GFS, grid_id별 행 단위)
def add_ws117_power_gfs(df):
    u10 = df["heightAboveGround_10_10u"]
    v10 = df["heightAboveGround_10_10v"]
    ws10 = np.sqrt(u10**2 + v10**2)

    u100 = df["heightAboveGround_100_100u"]
    v100 = df["heightAboveGround_100_100v"]
    ws100 = np.sqrt(u100**2 + v100**2)

    alpha = np.clip(np.log((ws100 + 1e-8) / (ws10 + 1e-8)) / np.log(100 / 10), -0.5, 1.0)
    df["ws117_power_gfs"] = ws100 * (117 / 100) ** alpha
    return df

# GFS 광역 공기 밀도 (grid_id별 행 단위)
def add_air_density_gfs(df):
    df = df.copy()
    df["air_density_gfs"] = df["surface_0_sp"] / (287.058 * df["heightAboveGround_2_2t"])
    out = df.groupby("forecast_kst_dtm", as_index=False)["air_density_gfs"].mean()
    return out

# GFS 광역 격자별 117m 보간 풍속 pivot (i = 1~9)
def add_ws117_gfs_grid(df):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    grid_wide = df.pivot(index="forecast_kst_dtm", columns="grid_id", values="ws117_power_gfs")
    grid_wide.columns = [f"ws117_gfs_grid_{int(g)}" for g in grid_wide.columns]
    return grid_wide.reset_index()

# GFS 전격자 117m 풍속 공간 평균 (시간당 1행, ws117_gfs_grid_{i} 필요)
def add_ws117_gfs_spatial_mean(df):
    grid_cols = [c for c in df.columns if c.startswith("ws117_gfs_grid_")]
    df["ws117_gfs_spatial_mean"] = df[grid_cols].mean(axis=1)
    return df

# GFS 광역 기압경도 (해면기압 격자간 차이, 시간당 1행)
def add_prmsl_range_gfs(df):
    out = df.groupby("forecast_kst_dtm")["meanSea_0_prmsl"].agg(lambda x: x.max() - x.min())
    out = out.rename("prmsl_range_gfs").reset_index()
    return out

# 117m 풍속 3승 (GFS, 시간당 1행, ws117_gfs_spatial_mean 필요)
def add_ws117_cube_gfs(df):
    df["ws117_cube_gfs"] = df["ws117_gfs_spatial_mean"] ** 3
    return df

# 바람 운동 에너지 플럭스 (GFS, 시간당 1행, air_density_gfs·ws117_cube_gfs 필요)
def add_wind_energy_flux_gfs(df):
    df["wind_energy_flux_gfs"] = 0.5 * df["air_density_gfs"] * df["ws117_cube_gfs"]
    return df

# 117m 풍속 1시간 차분 (GFS, 시간당 1행, ws117_gfs_spatial_mean 필요)
def add_ws117_diff_1h_gfs(df):
    df = df.sort_values("forecast_kst_dtm").reset_index(drop=True)
    df["ws117_diff_1h_gfs"] = df["ws117_gfs_spatial_mean"].diff(1)
    return df

# 24시간 전 예보 풍속 (GFS, 시간당 1행, ws117_gfs_spatial_mean 필요)
def add_ws117_lag_24h_gfs(df):
    df = df.sort_values("forecast_kst_dtm").reset_index(drop=True)
    df["ws117_lag_24h_gfs"] = df["ws117_gfs_spatial_mean"].shift(24)
    return df

# 6시간 롤링 이동평균 (GFS, 시간당 1행, ws117_gfs_spatial_mean 필요)
def add_ws117_roll_mean_6h_gfs(df):
    df = df.sort_values("forecast_kst_dtm").reset_index(drop=True)
    df["ws117_roll_mean_6h_gfs"] = df["ws117_gfs_spatial_mean"].rolling(window=6, min_periods=1).mean()
    return df

# 6시간 롤링 표준편차 (GFS, 시간당 1행, ws117_gfs_spatial_mean 필요)
def add_ws117_roll_std_6h_gfs(df):
    df = df.sort_values("forecast_kst_dtm").reset_index(drop=True)
    df["ws117_roll_std_6h_gfs"] = df["ws117_gfs_spatial_mean"].rolling(window=6, min_periods=2).std()
    return df



########## SCADA ##########
# UNISON 보정 풍향 
# (음수/360 초과 값을 0~360 범위로 보정, 1~5호기)
def add_unison_wd_360(scada_unison_train):
    df = scada_unison_train.copy()
    for i in range(1, 6):
        col = f"unison_wtg{i:02d}_wd"
        df[f"{col}_360"] = (df[col] + 360) % 360
    return df

# VESTAS Power 값 추정 
# (1~12호기 전체 통합, 풍속bin 중앙값 발전량 합산)
def add_vestas_power_curve_pred(scada_vestas_train, ws_bin_width=0.5, min_count=30):
    df = scada_vestas_train.copy()
    turbines = [f"vestas_wtg{i:02d}" for i in range(1, 13)]
    preds = []
    for wtg in turbines:
        power = df[f"{wtg}_power_kw10m"]
        ws_bin = np.floor(df[f"{wtg}_ws"] / ws_bin_width)
        temp = pd.DataFrame({"power": power, "ws_bin": ws_bin})
        median_val = temp.groupby("ws_bin")["power"].transform("median")
        count_val = temp.groupby("ws_bin")["power"].transform("count")
        preds.append(median_val.where(count_val >= min_count))
    df["vestas_power_curve_pred"] = pd.concat(preds, axis=1).sum(axis=1, min_count=1)
    return df

# UNISON Power 값 추정 
# (1~5호기 전체 통합, 풍속bin 중앙값 발전량 합산)
def add_unison_power_curve_pred(scada_unison_train, ws_bin_width=0.5, min_count=30):
    df = scada_unison_train.copy()
    turbines = [f"unison_wtg{i:02d}" for i in range(1, 6)]
    preds = []
    for wtg in turbines:
        power = df[f"{wtg}_power_kw10m"]
        ws_bin = np.floor(df[f"{wtg}_ws"] / ws_bin_width)
        temp = pd.DataFrame({"power": power, "ws_bin": ws_bin})
        median_val = temp.groupby("ws_bin")["power"].transform("median")
        count_val = temp.groupby("ws_bin")["power"].transform("count")
        preds.append(median_val.where(count_val >= min_count))
    df["unison_power_curve_pred"] = pd.concat(preds, axis=1).sum(axis=1, min_count=1)
    return df

# VESTAS group1 power값 추정 (1~6호기)
def add_vestas_power_curve_pred_group1(scada_vestas_train, ws_bin_width=0.5, min_count=30):
    df = scada_vestas_train.copy()
    turbines = [f"vestas_wtg{i:02d}" for i in range(1, 7)]
    preds = []
    for wtg in turbines:
        power = df[f"{wtg}_power_kw10m"]
        ws_bin = np.floor(df[f"{wtg}_ws"] / ws_bin_width)
        temp = pd.DataFrame({"power": power, "ws_bin": ws_bin})
        median_val = temp.groupby("ws_bin")["power"].transform("median")
        count_val = temp.groupby("ws_bin")["power"].transform("count")
        preds.append(median_val.where(count_val >= min_count))
    df["vestas_power_curve_pred_group1"] = pd.concat(preds, axis=1).sum(axis=1, min_count=1)
    return df

# VESTAS group2 power값 추정 (7~12호기)
def add_vestas_power_curve_pred_group2(scada_vestas_train, ws_bin_width=0.5, min_count=30):
    df = scada_vestas_train.copy()
    turbines = [f"vestas_wtg{i:02d}" for i in range(7, 13)]
    preds = []
    for wtg in turbines:
        power = df[f"{wtg}_power_kw10m"]
        ws_bin = np.floor(df[f"{wtg}_ws"] / ws_bin_width)
        temp = pd.DataFrame({"power": power, "ws_bin": ws_bin})
        median_val = temp.groupby("ws_bin")["power"].transform("median")
        count_val = temp.groupby("ws_bin")["power"].transform("count")
        preds.append(median_val.where(count_val >= min_count))
    df["vestas_power_curve_pred_group2"] = pd.concat(preds, axis=1).sum(axis=1, min_count=1)
    return df

# VESTAS group1 터빈별 상대 출력 계수 
# (1~6호기 E_i의 그룹 평균 1개 값)
def add_vestas_turbine_efficiency_group1(scada_vestas_train, ws_bin_width=0.5, eps=1e-8):
    df = scada_vestas_train.copy()
    turbines = [f"vestas_wtg{i:02d}" for i in range(1, 7)]
    turbine_eff = []
    for wtg in turbines:
        power = df[f"{wtg}_power_kw10m"]
        ws_bin = np.floor(df[f"{wtg}_ws"] / ws_bin_width)
        temp = pd.DataFrame({"power": power, "ws_bin": ws_bin})
        curve_pred = temp.groupby("ws_bin")["power"].transform("median")
        turbine_eff.append((curve_pred / (power + eps)).median())
    df["vestas_turbine_efficiency_group1"] = np.mean(turbine_eff)
    return df

# VESTAS group2 터빈별 상대 출력 계수 
# (7~12호기 E_i의 그룹 평균 1개 값)
def add_vestas_turbine_efficiency_group2(scada_vestas_train, ws_bin_width=0.5, eps=1e-8):
    df = scada_vestas_train.copy()
    turbines = [f"vestas_wtg{i:02d}" for i in range(7, 13)]
    turbine_eff = []
    for wtg in turbines:
        power = df[f"{wtg}_power_kw10m"]
        ws_bin = np.floor(df[f"{wtg}_ws"] / ws_bin_width)
        temp = pd.DataFrame({"power": power, "ws_bin": ws_bin})
        curve_pred = temp.groupby("ws_bin")["power"].transform("median")
        turbine_eff.append((curve_pred / (power + eps)).median())
    df["vestas_turbine_efficiency_group2"] = np.mean(turbine_eff)
    return df

# UNISON 터빈별 상대 출력 계수 
# (1~5호기 E_i의 그룹 평균 1개 값)
def add_unison_turbine_efficiency(scada_unison_train, ws_bin_width=0.5, eps=1e-8):
    df = scada_unison_train.copy()
    turbines = [f"unison_wtg{i:02d}" for i in range(1, 6)]
    turbine_eff = []
    for wtg in turbines:
        power = df[f"{wtg}_power_kw10m"]
        ws_bin = np.floor(df[f"{wtg}_ws"] / ws_bin_width)
        temp = pd.DataFrame({"power": power, "ws_bin": ws_bin})
        curve_pred = temp.groupby("ws_bin")["power"].transform("median")
        turbine_eff.append((curve_pred / (power + eps)).median())
    df["unison_turbine_efficiency"] = np.mean(turbine_eff)
    return df

# VESTAS group1 조건별 발전량 변동성 
# (1~6호기, 풍속·풍향 bin별 표준편차)
def add_vestas_power_variability_group1(scada_vestas_train, ws_bin_width=0.5, wd_bin_width=30, min_count=30):
    df = scada_vestas_train.copy()
    turbines = [f"vestas_wtg{i:02d}" for i in range(1, 7)]
    per_turbine = []
    for wtg in turbines:
        power = df[f"{wtg}_power_kw10m"]
        ws_bin = np.floor(df[f"{wtg}_ws"] / ws_bin_width)
        wd_bin = np.floor((df[f"{wtg}_wd"] % 360) / wd_bin_width)
        temp = pd.DataFrame({"power": power, "ws_bin": ws_bin, "wd_bin": wd_bin})
        std_val = temp.groupby(["ws_bin", "wd_bin"])["power"].transform("std")
        count_val = temp.groupby(["ws_bin", "wd_bin"])["power"].transform("count")
        per_turbine.append(std_val.where(count_val >= min_count))
    df["vestas_power_variability_group1"] = pd.concat(per_turbine, axis=1).mean(axis=1)
    return df

# VESTAS group2 조건별 발전량 변동성 
# (7~12호기, 풍속·풍향 bin별 표준편차)
def add_vestas_power_variability_group2(scada_vestas_train, ws_bin_width=0.5, wd_bin_width=30, min_count=30):
    df = scada_vestas_train.copy()
    turbines = [f"vestas_wtg{i:02d}" for i in range(7, 13)]
    per_turbine = []
    for wtg in turbines:
        power = df[f"{wtg}_power_kw10m"]
        ws_bin = np.floor(df[f"{wtg}_ws"] / ws_bin_width)
        wd_bin = np.floor((df[f"{wtg}_wd"] % 360) / wd_bin_width)
        temp = pd.DataFrame({"power": power, "ws_bin": ws_bin, "wd_bin": wd_bin})
        std_val = temp.groupby(["ws_bin", "wd_bin"])["power"].transform("std")
        count_val = temp.groupby(["ws_bin", "wd_bin"])["power"].transform("count")
        per_turbine.append(std_val.where(count_val >= min_count))
    df["vestas_power_variability_group2"] = pd.concat(per_turbine, axis=1).mean(axis=1)
    return df

# UNISON 터빈별 발전량 변동성 
# (1~5호기, 풍속·풍향 bin별 표준편차)
def add_unison_power_variability(scada_unison_train, ws_bin_width=0.5, wd_bin_width=30, min_count=30):
    df = scada_unison_train.copy()
    turbines = [f"unison_wtg{i:02d}" for i in range(1, 6)]
    per_turbine = []
    for wtg in turbines:
        power = df[f"{wtg}_power_kw10m"]
        ws_bin = np.floor(df[f"{wtg}_ws"] / ws_bin_width)
        wd_bin = np.floor((df[f"{wtg}_wd"] % 360) / wd_bin_width)
        temp = pd.DataFrame({"power": power, "ws_bin": ws_bin, "wd_bin": wd_bin})
        std_val = temp.groupby(["ws_bin", "wd_bin"])["power"].transform("std")
        count_val = temp.groupby(["ws_bin", "wd_bin"])["power"].transform("count")
        per_turbine.append(std_val.where(count_val >= min_count))
    df["unison_power_variability"] = pd.concat(per_turbine, axis=1).mean(axis=1)
    return df


########## 발전량 (이론적 파워커브) ##########
# group1 이론상의 파워커브 추정 발전량 (LDAPS 기준)
def add_power_curve_pred_group1_ldaps(df, grid_id):
    ws117 = df[f"ws117_ldaps_grid_{grid_id}"].to_numpy(dtype=float)
    pred = np.zeros_like(ws117)
    ramp = (ws117 >= 3) & (ws117 < 12.0)
    pred[ramp] = 21600 * ((ws117[ramp] - 3) / (12.0 - 3)) ** 3
    flat = (ws117 >= 12.0) & (ws117 < 22.5)
    pred[flat] = 21600
    df["power_curve_pred_group1_ldaps"] = pred
    return df

# group2 이론상의 파워커브 추정 발전량 (LDAPS 기준)
def add_power_curve_pred_group2_ldaps(df, grid_id):
    ws117 = df[f"ws117_ldaps_grid_{grid_id}"].to_numpy(dtype=float)
    pred = np.zeros_like(ws117)
    ramp = (ws117 >= 3) & (ws117 < 12.0)
    pred[ramp] = 21600 * ((ws117[ramp] - 3) / (12.0 - 3)) ** 3
    flat = (ws117 >= 12.0) & (ws117 < 22.5)
    pred[flat] = 21600
    df["power_curve_pred_group2_ldaps"] = pred
    return df

# group3 이론상의 파워커브 추정 발전량 (LDAPS 기준, UNISON 스펙)
def add_power_curve_pred_group3_ldaps(df, grid_id):
    ws117 = df[f"ws117_ldaps_grid_{grid_id}"].to_numpy(dtype=float)
    pred = np.zeros_like(ws117)
    ramp = (ws117 >= 3) & (ws117 < 12.5)
    pred[ramp] = 21000 * ((ws117[ramp] - 3) / (12.5 - 3)) ** 3
    flat = (ws117 >= 12.5) & (ws117 < 22.0)
    pred[flat] = 21000
    df["power_curve_pred_group3_ldaps"] = pred
    return df

# group1 이론상의 파워커브 추정 발전량 (GFS 기준)
def add_power_curve_pred_group1_gfs(df, grid_id):
    ws117 = df[f"ws117_gfs_grid_{grid_id}"].to_numpy(dtype=float)
    pred = np.zeros_like(ws117)
    ramp = (ws117 >= 3) & (ws117 < 12.0)
    pred[ramp] = 21600 * ((ws117[ramp] - 3) / (12.0 - 3)) ** 3
    flat = (ws117 >= 12.0) & (ws117 < 22.5)
    pred[flat] = 21600
    df["power_curve_pred_group1_gfs"] = pred
    return df

# group2 이론상의 파워커브 추정 발전량 (GFS 기준)
def add_power_curve_pred_group2_gfs(df, grid_id):
    ws117 = df[f"ws117_gfs_grid_{grid_id}"].to_numpy(dtype=float)
    pred = np.zeros_like(ws117)
    ramp = (ws117 >= 3) & (ws117 < 12.0)
    pred[ramp] = 21600 * ((ws117[ramp] - 3) / (12.0 - 3)) ** 3
    flat = (ws117 >= 12.0) & (ws117 < 22.5)
    pred[flat] = 21600
    df["power_curve_pred_group2_gfs"] = pred
    return df

# group3 이론상의 파워커브 추정 발전량 (GFS 기준, UNISON 스펙)
def add_power_curve_pred_group3_gfs(df, grid_id):
    ws117 = df[f"ws117_gfs_grid_{grid_id}"].to_numpy(dtype=float)
    pred = np.zeros_like(ws117)
    ramp = (ws117 >= 3) & (ws117 < 12.5)
    pred[ramp] = 21000 * ((ws117[ramp] - 3) / (12.5 - 3)) ** 3
    flat = (ws117 >= 12.5) & (ws117 < 22.0)
    pred[flat] = 21000
    df["power_curve_pred_group3_gfs"] = pred
    return df


########## 기타-기상예보 ##########
# 풍향 순환 인코딩(cos)
def add_wd_cos(df):
    u100 = df['heightAboveGround_100_100u']
    v100 = df['heightAboveGround_100_100v']
    
    wind_dir = np.arctan2(-u100, -v100)
    df['wd_cos'] = np.cos(wind_dir)
    return df

# 풍향 순환 인코딩(sin)
def add_wd_sin(df):
    u100 = df['heightAboveGround_100_100u']
    v100 = df['heightAboveGround_100_100v']
    
    wind_dir = np.arctan2(-u100, -v100)
    df['wd_sin'] = np.sin(wind_dir)
    return df

# 대기 난류강도 추정치
def add_turbulence_intensity(df):
    u10 = df['heightAboveGround_10_10u']
    v10 = df['heightAboveGround_10_10v']
    ws10 = np.sqrt(u10**2 + v10**2)
    
    df['turbulence_intensity'] = (df['surface_0_gust'] - ws10) / (ws10 + 1e-8)
    return df

# 연중 주기(cos)
def add_dayofyear_cos(df):
    dt = pd.to_datetime(df['forecast_kst_dtm'])
    df['dayofyear_cos'] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
    return df

# 연중 주기(sin)
def add_dayofyear_sin(df):
    dt = pd.to_datetime(df['forecast_kst_dtm'])
    df['dayofyear_sin'] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
    return df

# 일중 주기(cos)
def add_hour_cos(df):
    dt = pd.to_datetime(df['forecast_kst_dtm'])
    df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    return df

# 일중 주기(sin)
def add_hour_sin(df):
    dt = pd.to_datetime(df['forecast_kst_dtm'])
    df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    return df

# 117m 직교 방향 풍속
def add_ws117_channel_cross(df):
    u10 = df['heightAboveGround_10_10u']
    v10 = df['heightAboveGround_10_10v']
    ws10 = np.sqrt(u10**2 + v10**2)
    
    u100 = df['heightAboveGround_100_100u']
    v100 = df['heightAboveGround_100_100v']
    ws100 = np.sqrt(u100**2 + v100**2)
    
    alpha = np.log(ws100 / (ws10 + 1e-8)) / np.log(100/10)
    
    u117 = u100 * ((117/100) ** alpha)
    v117 = v100 * ((117/100) ** alpha)
    
    # Group 1, 2 (VESTAS): 270도
    theta_g12 = np.radians(270)
    df['ws117_channel_cross_g12'] = u117 * np.cos(theta_g12) - v117 * np.sin(theta_g12)
    
    # Group 3 (UNISON): 45도
    theta_g3 = np.radians(45)
    df['ws117_channel_cross_g3'] = u117 * np.cos(theta_g3) - v117 * np.sin(theta_g3)
    
    return df

# 117m 채널축 방향 풍속
def add_ws117_channel_along(df):
    u10 = df['heightAboveGround_10_10u']
    v10 = df['heightAboveGround_10_10v']
    ws10 = np.sqrt(u10**2 + v10**2)
    
    u100 = df['heightAboveGround_100_100u']
    v100 = df['heightAboveGround_100_100v']
    ws100 = np.sqrt(u100**2 + v100**2)
    
    alpha = np.log(ws100 / (ws10 + 1e-8)) / np.log(100/10)
    
    u117 = u100 * ((117/100) ** alpha)
    v117 = v100 * ((117/100) ** alpha)
    
    # Group 1, 2 (VESTAS): 270도
    theta_g12 = np.radians(270)
    df['ws117_channel_along_g12'] = u117 * np.sin(theta_g12) + v117 * np.cos(theta_g12)
    
    # Group 3 (UNISON): 45도
    theta_g3 = np.radians(45)
    df['ws117_channel_along_g3'] = u117 * np.sin(theta_g3) + v117 * np.cos(theta_g3)
    
    return df
