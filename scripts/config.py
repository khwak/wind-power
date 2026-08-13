import os
from pathlib import Path

##### 기본 설정 ##### 

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

# 각 그룹의 라벨 제공기간(data_description.md 기준: group1/2=2022~2024, group3=2023~2024) 안에서
# 4계절을 고르게 훑도록 분기별 fold 시작일을 잡는다.
# 실행용
# SEASONAL_CV_FOLD_STARTS = {
#     "kpx_group_1": ["2022-04-01", "2022-07-01", "2022-10-01", "2023-01-01",
#                      "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"],
#     "kpx_group_2": ["2022-04-01", "2022-07-01", "2022-10-01", "2023-01-01",
#                      "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"],
#     "kpx_group_3": ["2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"],
# }
# offline용
SEASONAL_CV_FOLD_STARTS = {
    "kpx_group_1": ["2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"],
    "kpx_group_2": ["2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"],
    "kpx_group_3": ["2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"],
}
CV_WINDOW_DAYS = 90  # fold 하나가 담당하는 검증 구간 길이 (~한 계절)


##### 변수 중요도 분석 기반 제외 feature #####
# 주의: feature_importance CSV는 XGB/LGBM 중요도만 반영한다. curve(Ridge)는 별도 계수를 쓰므로
# 여기 중요도가 안 잡힌다. ws117_cube_ldaps, wind_energy_flux_ldaps는 트리 중요도는 낮지만
# Ridge가 3승 물리관계를 선형으로 근사하는 데 구조적으로 의존하는 feature라 절대 제외하지 않는다.
EXCLUDED_FEATURES = [
    # LDAPS 격자별 풍향(wd117)
    "wd117_ldaps_grid_1", "wd117_ldaps_grid_2", "wd117_ldaps_grid_3", "wd117_ldaps_grid_4",
    "wd117_ldaps_grid_5", "wd117_ldaps_grid_6", "wd117_ldaps_grid_7", "wd117_ldaps_grid_8",
    "wd117_ldaps_grid_9", "wd117_ldaps_grid_10", "wd117_ldaps_grid_11", "wd117_ldaps_grid_12",
    "wd117_ldaps_grid_13", "wd117_ldaps_grid_14", "wd117_ldaps_grid_15", "wd117_ldaps_grid_16",
    # LDAPS 격자별 풍속(ws117)
    "ws117_ldaps_grid_1", "ws117_ldaps_grid_2", "ws117_ldaps_grid_3", "ws117_ldaps_grid_4",
    "ws117_ldaps_grid_5", "ws117_ldaps_grid_6", "ws117_ldaps_grid_7", "ws117_ldaps_grid_8",
    "ws117_ldaps_grid_10", "ws117_ldaps_grid_11", "ws117_ldaps_grid_12",
    "ws117_ldaps_grid_14", "ws117_ldaps_grid_15", "ws117_ldaps_grid_16",
]


###### 하이퍼파라미터 설정 #####

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
    "n_estimators": 800,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1
}

LGBM_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "min_child_weight": 1.0,   
    "reg_lambda": 1.0,   
    "verbosity": -1,      
}

# 손실함수 커스텀 설정
LOSS_TYPES = ["huber_capacity", "threshold_weighted", "smooth_ficr", "threshold_weighted_huber"]
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


# 곡선 모델 하이퍼파라미터
# 수정 (추가) — 발견 단계와 실행 단계가 공유하는 설정
REGIME_CONFIG_PATH = MODEL_DIR / "regime_config.json"

# discover_regimes.py 전용 설정
REGIME_WS_BIN_WIDTH = 1.0
REGIME_MIN_BIN_SAMPLES = 100         
REGIME_MIN_FOLD_BIN_SAMPLES = 30    
REGIME_MIN_FOLD_AGREE_RATIO = {
    "kpx_group_1": 0.75,
    "kpx_group_2": 0.75,
    "kpx_group_3": 0.5,   
}  
REGIME_MIN_WIDTH = {
    "kpx_group_1": 2.0,
    "kpx_group_2": 2.0,
    "kpx_group_3": 1.0,   
}
RIDGE_PARAMS = {"alpha": 1.0, "random_state": 42}

MLP_PARAMS = {
    "hidden_dims": (128, 64, 32),
    "lr": 1e-3,
    "max_epochs": 300,
    "patience": 20,
    "batch_size": 512,
    "k": 30.0,       # differentiable FICR loss의 경계선 뾰족한 정도 (threshold_weighted의 sigma 역할)
    "warmup_epochs": 15,
    "seed": 42,
}

# GAM에서 스플라인(곡선)으로 적합할 물리적 파워커브 관련 변수들 (그룹별로 다름)
# prepare_data_all.py 버전
# GAM_SPLINE_COLS = {
#     "kpx_group_1": [
#         "ws117_ldaps_spatial_mean", "ws117_gfs_spatial_mean",
#         "wind_energy_flux_ldaps", "wind_energy_flux_gfs",
#         "power_curve_pred_group1_ldaps", "power_curve_pred_group1_gfs",
#         "power_curve_pred_lookup_group1_gfs", "vestas_power_curve_pred_group1",
#     ],
#     "kpx_group_2": [
#         "ws117_ldaps_spatial_mean", "ws117_gfs_spatial_mean",
#         "wind_energy_flux_ldaps", "wind_energy_flux_gfs",
#         "power_curve_pred_group2_ldaps", "power_curve_pred_group2_gfs",
#         "power_curve_pred_lookup_group2_gfs", "vestas_power_curve_pred_group2",
#     ],
#     "kpx_group_3": [
#         "ws117_ldaps_spatial_mean", "ws117_gfs_spatial_mean",
#         "wind_energy_flux_ldaps", "wind_energy_flux_gfs",
#         "power_curve_pred_group3_ldaps", "power_curve_pred_group3_gfs",
#         "unison_power_curve_pred",
#     ],
# }

# GAM에서 data leakage 반영 후 버전.
GAM_SPLINE_COLS = {
    "kpx_group_1": [
        "ws117_ldaps_spatial_mean", 
        "wind_energy_flux_ldaps", 
        "vestas_power_curve_pred_group1",
    ],
    "kpx_group_2": [
        "ws117_ldaps_spatial_mean", 
        "wind_energy_flux_ldaps", 
        "vestas_power_curve_pred_group2",
    ],
    "kpx_group_3": [
        "ws117_ldaps_spatial_mean", 
        "wind_energy_flux_ldaps", 
        "unison_power_curve_pred",
    ],
}
GAM_PARAMS = {"n_splines": 15, "lam": 0.6}


##### 공통 풍속 기준 컬럼 #####
# 격자 선택 방식이 바뀌면 수정 필요
# WS_FEATURE_COL = "ws117_ldaps_grid_13"
WS_FEATURE_COL = {
    "kpx_group_1": "ws117_calibrated_group1",
    "kpx_group_2": "ws117_calibrated_group2",
    "kpx_group_3": "ws117_calibrated_group3",
}

# 풍속구간별 샘플 가중치
# regime_config.json에서 확인된 그룹별 램프구간
# (잠정치, WS_FEATURE_COL 기준 — 격자 선택 방식이 바뀌면 재확인 필요)
RAMP_WS_RANGES = {
    "kpx_group_1": (6.0, 12.0),  
    "kpx_group_2": (6.0, 12.0),  
    "kpx_group_3": (6.0, 12.5),  
}
RAMP_SAMPLE_WEIGHT = 3.5  # 램프구간 표본에 곱할 가중치 배수. 처음엔 보수적으로 2.0부터


##### 격자 선택 & shear 보정 (팀원 분석 반영용) #####

# "nearest": 그룹 중심 최근접 격자 / "correlation": cutoff마다 재계산 (fold별로 흔들릴 수 있음)
# "manual" : 검증 완료된 고정값 사용 (권장 — 팀원 분석 + 유효구간 + 계절별 검증 결과 반영)
GRID_SELECTION_METHOD = "manual"

# method="manual"일 때 사용할 고정 격자.
# 근거: 팀원 상관관계 분석(전체구간) + 유효구간 재검증 + 4계절 검증(봄/가을/겨울 뚜렷 1위,
# 여름은 group2/3 grid8과 근소한 차이) — 종합적으로 grid13이 3그룹 전부에서 안정적 1위.
GRID_MANUAL_SELECTION = {1: 13, 2: 13, 3: 13}

# 그룹별 shear 외삽 보정. {group: {"scale": a, "offset": b}} 형태.
# 비어있으면(={}) 보정 미적용 = 기존 동작과 100% 동일.
SHEAR_BIAS_CORRECTION = {}
# 계절별 shear 보정 — 계절 클러스터(seasonal_clusters.json) 하나당 최소 표본 수.

SHEAR_MIN_SEASON_SAMPLES = 300

##### 격자 간 gradient feature #####
# partial correlation 분석(grid13 통제 후 편상관) 결과 상위였던 쌍부터 우선 반영
# 각 튜플은 (a, b) → 새 feature명: ws117_ldaps_diff_{a}_{b} = grid_a - grid_b
GRID_DIFF_PAIRS = [(13, 9)]


##### 분위수 기반 전략적 예측 (GEFCom2014 winner 방식) #####
QUANTILE_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
QUANTILE_CALIBRATION_RATIO = 0.20  # X_tr 꼬리에서 보정용 비율

# 기대정산금 최대화 탐색 시 후보값 격자 설정
QUANTILE_SEARCH_MARGIN_RATIO = 0.08   # min/max 분위수 예측값 양옆으로 이만큼(비율) 더 넓게 탐색
QUANTILE_SEARCH_N_CANDIDATES = 121    # 후보값 개수 (많을수록 정밀, 느려짐)

LGBM_QUANTILE_PARAMS = {
    "n_estimators": 1500,       
    "learning_rate": 0.02,       
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "verbosity": -1,
}

XGB_QUANTILE_PARAMS = {
    "n_estimators": 1500,        
    "learning_rate": 0.02,       
    "max_depth": 4,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}




###### Optuna 하이퍼파라미터 #####

OPTUNA_N_TRIALS = 60          # 그룹x모델 조합당 시행 횟수
OPTUNA_SEED = 42
HUBER_HESS_FLOOR = 1e-2

# 다중 fold Optuna 탐색용 trial 예산 
# SEASONAL_OPTUNA_TRIAL_BUDGET = {"huber_capacity": 10, "threshold_weighted": 15, "smooth_ficr": 15}
# SEASONAL_OPTUNA_TRIAL_BUDGET = {"huber_capacity": 10, "threshold_weighted": 15}
# SEASONAL_OPTUNA_TRIAL_BUDGET = {"huber_capacity": 10, "threshold_weighted": 10, "threshold_weighted_huber": 20}
SEASONAL_OPTUNA_TRIAL_BUDGET = {"huber_capacity": 15, "threshold_weighted": 25, "huber_threshold": 25}

# 손실함수 탐색 공간
OPTUNA_SEARCH_SPACE = {
    "huber_capacity": {
        "delta": {"low": 0.05, "high": 0.30, "log": True},
    },
    "threshold_weighted": {
        "amplitude": {"low": 0.1, "high": 15.0, "log": True},
        "sigma": {"low": 0.001, "high": 0.05, "log": True},
    },
    "smooth_ficr": {
        "anchor_weight": {"low": 0.5, "high": 0.95, "log": False},  # 0.1~0.999 → 0.5~0.95
        "k": {"low": 50, "high": 800, "log": True},
    },
    "threshold_weighted_huber": {
        "delta": {"low": 0.05, "high": 0.30, "log": True},
        "amplitude": {"low": 0.5, "high": 10.0, "log": True},
        "sigma": {"low": 0.002, "high": 0.03, "log": True},
    },
    "huber_threshold": {                                   
        "delta": {"low": 0.05, "high": 0.30, "log": True},   
        "amplitude": {"low": 0.1, "high": 15.0, "log": True},
        "sigma": {"low": 0.001, "high": 0.05, "log": True},
    },
}

STABILITY_LAMBDA = 0.0

# 모델 탐색 공간
XGB_SEARCH_SPACE = {
    "n_estimators": {"low": 100, "high": 800, "log": False, "type": "int"},
    "max_depth": {"low": 3, "high": 7, "log": False, "type": "int"},              # [개선] 10 -> 7 축소 (과적합 방지)
    "learning_rate": {"low": 0.01, "high": 0.08, "log": True, "type": "float"},   # [개선] 0.2 -> 0.08 축소
    "subsample": {"low": 0.6, "high": 1.0, "log": False, "type": "float"},
    "colsample_bytree": {"low": 0.5, "high": 1.0, "log": False, "type": "float"},
    "reg_lambda": {"low": 1.0, "high": 10.0, "log": True, "type": "float"},       # [신규] L2 정규화 페널티 추가
}
LGBM_SEARCH_SPACE = {
    "n_estimators": {"low": 100, "high": 800, "log": False, "type": "int"},
    "num_leaves": {"low": 15, "high": 127, "log": False, "type": "int"},           # [개선] 127 -> 63 축소 (과적합 방지)
    "learning_rate": {"low": 0.01, "high": 0.08, "log": True, "type": "float"},   # [개선] 0.2 -> 0.08 축소
    "subsample": {"low": 0.6, "high": 1.0, "log": False, "type": "float"},
    "colsample_bytree": {"low": 0.5, "high": 1.0, "log": False, "type": "float"},
    "reg_lambda": {"low": 1.0, "high": 10.0, "log": True, "type": "float"},       # [신규] L2 정규화 페널티 추가
}

##### final 버전 - 최적화 하드코딩 #####
# BEST_MODEL_CONFIG = {
#     "kpx_group_1": {
#         "XGB":  {"random_state": 42, "n_jobs": -1, "n_estimators": 707, "max_depth": 6, "learning_rate": 0.04359666365651395, "subsample": 0.608233797718321, "colsample_bytree": 0.9849549260809971, "reg_lambda": 6.798962421591129},
#         "LGBM": {"random_state": 42, "min_child_weight": 1.0, "verbosity": -1, "importance_type": "gain", "n_estimators": 711, "num_leaves": 64, "learning_rate": 0.0796367226999673, "subsample": 0.8509679306459871, "colsample_bytree": 0.666412081433084, "reg_lambda": 9.895615387708215},
#     },
#     "kpx_group_2": {
#         "XGB":  {"random_state": 42, "n_jobs": -1, "n_estimators": 402, "max_depth": 4, "learning_rate": 0.035690959437712715, "subsample": 0.6557975442608167, "colsample_bytree": 0.6460723242676091, "reg_lambda": 2.324672848950434},
#         "LGBM": {"random_state": 42, "min_child_weight": 1.0, "verbosity": -1, "importance_type": "gain", "n_estimators": 402, "num_leaves": 47, "learning_rate": 0.035690959437712715, "subsample": 0.6557975442608167, "colsample_bytree": 0.6460723242676091, "reg_lambda": 2.324672848950434},
#     },
#     "kpx_group_3": {
#         "XGB":  {"random_state": 42, "n_jobs": -1, "n_estimators": 753, "max_depth": 6, "learning_rate": 0.034690153706107965, "subsample": 0.7691005300077898, "colsample_bytree": 0.7472234962527935, "reg_lambda": 1.2011881007835652},
#         "LGBM": {"random_state": 42, "min_child_weight": 1.0, "verbosity": -1, "importance_type": "gain", "n_estimators": 317, "num_leaves": 51, "learning_rate": 0.04559319574629467, "subsample": 0.8550229885420852, "colsample_bytree": 0.9436063712881633, "reg_lambda": 2.9662989987000667},
#     },
# }

# test
BEST_MODEL_CONFIG = {
    "kpx_group_1": {
        "XGB":  {"random_state": 42, "n_jobs": -1, "n_estimators": 499, "max_depth": 4, "learning_rate": 0.04195620299074925, "subsample": 0.7424364893339975, "colsample_bytree": 0.720932341890189, "reg_lambda": 4.542425989748594},
        "LGBM": {"random_state": 42, "min_child_weight": 1.0, "verbosity": -1, "importance_type": "gain", "n_estimators": 660, "num_leaves": 68, "learning_rate": 0.059750451948554786, "subsample": 0.6023899357045669, "colsample_bytree": 0.8199638193192597, "reg_lambda": 2.9318471869411504},
    },
    "kpx_group_2": {
        "XGB":  {"random_state": 42, "n_jobs": -1, "n_estimators": 469, "max_depth": 5, "learning_rate": 0.06249944751725576, "subsample": 0.9031934126795159, "colsample_bytree": 0.7426604485466712, "reg_lambda": 1.0622600189983782},
        "LGBM": {"random_state": 42, "min_child_weight": 1.0, "verbosity": -1, "importance_type": "gain", "n_estimators": 776, "num_leaves": 106, "learning_rate": 0.018840552955192824, "subsample": 0.6390688456025535, "colsample_bytree": 0.8421165132560784, "reg_lambda": 2.7551959649510764},
    },
    "kpx_group_3": {
        "XGB":  {"random_state": 42, "n_jobs": -1, "n_estimators": 228, "max_depth": 4, "learning_rate": 0.02977846306223348, "subsample": 0.7727780074568463, "colsample_bytree": 0.645614570099021, "reg_lambda": 4.091220574443785},
        "LGBM": {"random_state": 42, "min_child_weight": 1.0, "verbosity": -1, "importance_type": "gain", "n_estimators": 448, "num_leaves": 33, "learning_rate": 0.04561487895116297, "subsample": 0.8468838671697153, "colsample_bytree": 0.728920203341694, "reg_lambda": 1.727665941743956},
    },
}

# Baseline
# BEST_LOSS_CONFIG = {
#     "kpx_group_1": {
#         "XGB":  {"loss_name": "threshold_weighted", "params": {"amplitude": 5.847938552236825, "sigma": 0.005485844105337834}},
#         "LGBM": {"loss_name": "huber_capacity", "params": {"delta": 0.13316598070397054}},
#     },
#     "kpx_group_2": {
#         "XGB":  {"loss_name": "threshold_weighted", "params": {"amplitude": 3.302597152071603, "sigma": 0.021198402317468616}},
#         "LGBM": {"loss_name": "threshold_weighted", "params": {"amplitude": 2.3778501069975473, "sigma": 0.0168655540161967}},
#     },
#     "kpx_group_3": {
#         "XGB":  {"loss_name": "huber_capacity", "params": {"delta": 0.14595756869825136}},        # smooth_ficr → huber_capacity로 교체
#         "LGBM": {"loss_name": "threshold_weighted", "params": {"amplitude": 2.6315085506167315, "sigma": 0.00296053555003479}},  # smooth_ficr → threshold_weighted로 교체
#     },
# }
# BEST_LOSS_CONFIG = {
#     "kpx_group_1": {
#         "XGB":  {"loss_name": "huber_capacity", "params": {"delta": 0.0554840098004973}},
#         "LGBM": {"loss_name": "threshold_weighted_huber", "params": {"delta": 0.29468127479916906, "amplitude": 4.797742370763172, "sigma": 0.0033431269538905647}},
#     },
#     "kpx_group_2": {
#         "XGB":  {"loss_name": "threshold_weighted", "params": {"amplitude": 0.4592602792418462, "sigma": 0.0077901431262762414}},
#         "LGBM": {"loss_name": "threshold_weighted", "params": {"amplitude": 0.4592602792418462, "sigma": 0.0077901431262762414}},
#     },
#     "kpx_group_3": {
#         "XGB":  {"loss_name": "threshold_weighted_huber", "params": {"delta": 0.2805674379379642, "amplitude": 3.72422814524701, "sigma": 0.00429474219228544}},
#         "LGBM": {"loss_name": "threshold_weighted_huber", "params": {"delta": 0.15275316627633723, "amplitude": 1.3473432786208845, "sigma": 0.002375638833476553}},
#     },
# }

# BEST_LOSS_CONFIG = {
#     "kpx_group_1": {
#         "XGB":  {"loss_name": "huber_threshold", "params": {"delta": 0.06312747042309949, "amplitude": 3.1959950475848333, "sigma": 0.0015986255106052408}},
#         "LGBM": {"loss_name": "huber_threshold", "params": {"delta": 0.21492316324835245, "amplitude": 2.7546877614333125, "sigma": 0.004220720624548675}},
#     },
#     "kpx_group_2": {
#         "XGB":  {"loss_name": "huber_threshold", "params": {"delta": 0.09070690911895868, "amplitude": 0.3183172090788175, "sigma": 0.006254136716066883}},
#         "LGBM": {"loss_name": "huber_threshold", "params": {"delta": 0.11569593130196097, "amplitude": 0.1931331178011487, "sigma": 0.005456986232932154}},
#     },
#     "kpx_group_3": {
#         "XGB":  {"loss_name": "huber_threshold", "params": {"delta": 0.08558956910286954, "amplitude": 0.3864086696159432, "sigma": 0.011120357873223699}},
#         "LGBM": {"loss_name": "huber_threshold", "params": {"delta": 0.21140199214218833, "amplitude": 0.8931856467073521, "sigma": 0.015372469604656767}},
#     },
# }

# test
BEST_LOSS_CONFIG = {
    "kpx_group_1": {
        "XGB":  {"loss_name": "huber_capacity", "params": {"delta": 0.05101472208866336}},
        "LGBM": {"loss_name": "huber_threshold", "params": {"delta": 0.15583931767075299, "amplitude": 0.7805498896204464, "sigma": 0.013707341249762393}},
    },
    "kpx_group_2": {
        "XGB":  {"loss_name": "threshold_weighted", "params": {"amplitude": 2.9697158814524376, "sigma": 0.003542796928018529}},
        "LGBM": {"loss_name": "threshold_weighted", "params": {"amplitude": 0.13853458335022473, "sigma": 0.04093813608598782}},
    },
    "kpx_group_3": {
        "XGB":  {"loss_name": "huber_capacity", "params": {"delta": 0.06925598810528834}},
        "LGBM": {"loss_name": "threshold_weighted", "params": {"amplitude": 3.775953662454377, "sigma": 0.013400087150160007}},
    },
}


# Chronos 모델 설정 (Chronos-2 기반)
# CHRONOS_MODEL_PATH = "amazon/chronos-bolt-small"
# CHRONOS_MODEL_PATH = "amazon/chronos-2"
CHRONOS_MODEL_PATH = "autogluon/chronos-2"