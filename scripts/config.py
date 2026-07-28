import os
from pathlib import Path

# 기본 디렉토리 설정
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

for d in [PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 타겟 및 용량 설정
TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {
    "kpx_group_1": 21600,
    "kpx_group_2": 21600,
    "kpx_group_3": 21000,
}

# 그룹별 터빈 파워커브 스펙
TURBINE_SPEC = {
    "kpx_group_1": {"capacity": 21600, "v_in": 3.0, "v_rated": 12.0, "v_out": 22.5},  # VESTAS
    "kpx_group_2": {"capacity": 21600, "v_in": 3.0, "v_rated": 12.0, "v_out": 22.5},  # VESTAS
    "kpx_group_3": {"capacity": 21000, "v_in": 3.0, "v_rated": 12.5, "v_out": 22.0},  # UNISON
}

# 지형 채널축 각도 (ws117_channel_along / cross 산출용, 단위: degree)
CHANNEL_AXIS_DEG = {
    "main": 67.5,   # 주축: WSW <-> NE 골짜기 방향
    "alt": 157.5,   # 보조축: 주축과 직교 (67.5 + 90), 검증용 보조 피처
}


######

# 손실함수/평가지표 커스텀 관련 설정
VALID_RATIO_THRESHOLD = 0.10    
EXCLUDE_INVALID_ROWS = True     
INVALID_SAMPLE_WEIGHT = 0.1     
VAL_HOLDOUT_RATIO = 0.10        
EARLY_STOPPING_ROUNDS = 50

# Ensemble 하이퍼파라미터
RF_PARAMS = {
    "n_estimators": 120,
    "max_depth": 14,
    "min_samples_leaf": 8,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

ET_PARAMS = {
    "n_estimators": 300, 
    "random_state": 42, 
    "n_jobs": -1
}

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1
}

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

# Chronos 모델 설정 (Chronos-2 기반)
# CHRONOS_MODEL_PATH = "amazon/chronos-bolt-small"
# CHRONOS_MODEL_PATH = "amazon/chronos-2"
CHRONOS_MODEL_PATH = "autogluon/chronos-2"

# 손실함수 커스텀 설정
LOSS_TYPES = ["huber_capacity", "threshold_weighted", "smooth_ficr"]
LOSS_PARAM_GRIDS = {
    "huber_capacity": [
        {"delta": d} for d in [0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    ],
    "threshold_weighted": [
        {"amplitude": a, "sigma": s}
        for a in [0.5, 1.0, 3.0, 5.0, 10.0]
        for s in [0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04]
    ],
    "smooth_ficr": [
        {"anchor_weight": aw, "k": k}
        for aw in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        for k in [100, 300, 500]
    ],
}

HUBER_HESS_FLOOR = 1e-2

# Optuna 하이퍼파라미터
OPTUNA_N_TRIALS = 60          # 그룹x모델 조합당 시행 횟수
OPTUNA_SEED = 42
HUBER_HESS_FLOOR = 1e-2

OPTUNA_SEARCH_SPACE = {
    "huber_capacity": {
        "delta": {"low": 0.01, "high": 0.30, "log": True},
    },
    "threshold_weighted": {
        "amplitude": {"low": 0.1, "high": 15.0, "log": True},
        "sigma": {"low": 0.001, "high": 0.05, "log": True},
    },
    "smooth_ficr": {
        "anchor_weight": {"low": 0.1, "high": 0.999, "log": False},
        "k": {"low": 50, "high": 800, "log": True},
    },
}

# 최적화 하드코딩
BEST_LOSS_CONFIG = {
    "kpx_group_1": {
        "XGB":  {"loss_name": "threshold_weighted", "params": {"amplitude": 0.9150143222922885, "sigma": 0.02324908768223826}},
        "LGBM": {"loss_name": "threshold_weighted", "params": {"amplitude": 2.2993058417334544, "sigma": 0.03728367815703181}},
    },
    "kpx_group_2": {
        "XGB":  {"loss_name": "threshold_weighted", "params": {"amplitude": 0.5, "sigma": 0.003}},
        "LGBM": {"loss_name": "threshold_weighted", "params": {"amplitude": 1.0, "sigma": 0.04}},
    },
    "kpx_group_3": {
        "XGB":  {"loss_name": "smooth_ficr", "params": {"anchor_weight": 0.9879497516800868, "k": 148}},
        "LGBM": {"loss_name": "threshold_weighted", "params": {"amplitude": 6.478450641951017, "sigma": 0.002294868368113055}},
    },
}