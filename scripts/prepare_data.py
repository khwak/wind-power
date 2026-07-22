import pandas as pd
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

def get_tabular_data():
    """학습 및 테스트용 Tabular 데이터셋 병합 및 반환"""
    # 데이터 로드
    train_labels = pd.read_csv(RAW_DIR / "train" / "train_labels.csv", encoding="utf-8-sig")
    sample_sub = pd.read_csv(RAW_DIR / "sample_submission.csv", encoding="utf-8-sig")
    
    ldaps_train = pd.read_csv(RAW_DIR / "train" / "ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv(RAW_DIR / "train" / "gfs_train.csv", encoding="utf-8-sig")
    ldaps_test = pd.read_csv(RAW_DIR / "test" / "ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv(RAW_DIR / "test" / "gfs_test.csv", encoding="utf-8-sig")

    train_labels["kst_dtm"] = pd.to_datetime(train_labels["kst_dtm"])
    sample_sub["forecast_kst_dtm"] = pd.to_datetime(sample_sub["forecast_kst_dtm"])

    # 날씨 데이터 병합
    train_weather = aggregate_weather(ldaps_train, "ldaps").merge(
        aggregate_weather(gfs_train, "gfs"), on="forecast_kst_dtm", how="inner"
    )
    test_weather = aggregate_weather(ldaps_test, "ldaps").merge(
        aggregate_weather(gfs_test, "gfs"), on="forecast_kst_dtm", how="inner"
    )

    train_base = train_labels.rename(columns={"kst_dtm": "forecast_kst_dtm"})
    train_df = train_base.merge(train_weather, on="forecast_kst_dtm", how="left")
    test_df = sample_sub[["forecast_id", "forecast_kst_dtm"]].merge(
        test_weather, on="forecast_kst_dtm", how="left"
    )

    X_train = pd.concat([
        calendar_features(train_df["forecast_kst_dtm"]), 
        train_df.drop(columns=["forecast_kst_dtm", *TARGET_COLS])
    ], axis=1)
    
    X_test = pd.concat([
        calendar_features(test_df["forecast_kst_dtm"]), 
        test_df.drop(columns=["forecast_id", "forecast_kst_dtm"])
    ], axis=1)

    return train_df, X_train, test_df, X_test, sample_sub