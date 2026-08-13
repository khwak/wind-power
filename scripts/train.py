import argparse
import numpy as np
import pandas as pd
import json
import datetime
import optuna
import lightgbm as lgb
import torch
import torch.nn as nn
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pygam import LinearGAM, s, l
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


from config import (
    TARGET_COLS, CAPACITY_KWH, RF_PARAMS, ET_PARAMS, XGB_PARAMS, LGBM_PARAMS, OUTPUT_DIR, MODEL_DIR,
    VALID_RATIO_THRESHOLD, EXCLUDE_INVALID_ROWS, INVALID_SAMPLE_WEIGHT,
    VAL_HOLDOUT_RATIO, EARLY_STOPPING_ROUNDS,
    OPTUNA_N_TRIALS, OPTUNA_SEED, OPTUNA_SEARCH_SPACE, HUBER_HESS_FLOOR,
    BEST_LOSS_CONFIG, XGB_SEARCH_SPACE, LGBM_SEARCH_SPACE, BEST_MODEL_CONFIG,
    RIDGE_PARAMS, MLP_PARAMS, GAM_SPLINE_COLS, GAM_PARAMS,
    REGIME_CONFIG_PATH, CV_WINDOW_DAYS,
    WS_FEATURE_COL, RAMP_WS_RANGES, RAMP_SAMPLE_WEIGHT, SEASONAL_OPTUNA_TRIAL_BUDGET,
    QUANTILE_LEVELS, QUANTILE_SEARCH_MARGIN_RATIO, QUANTILE_SEARCH_N_CANDIDATES, QUANTILE_CALIBRATION_RATIO,
    LGBM_QUANTILE_PARAMS, XGB_QUANTILE_PARAMS, STABILITY_LAMBDA,
)
from prepare_data import get_tabular_data, get_target_xy, get_bounded_validation_xy

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================
# 대회 공식 FICR 관련 함수 (기존과 동일, 유지)
# ============================================
def _neg_ficr(y_true, y_pred, capacity):
    valid = y_true >= capacity * VALID_RATIO_THRESHOLD
    if valid.sum() == 0:
        return 0.0
    error_rate = np.abs(y_pred[valid] - y_true[valid]) / capacity
    unit_price = np.select([error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0)
    ficr = np.sum(y_true[valid] * unit_price) / np.sum(y_true[valid] * 4.0)
    return -ficr


def _lgb_ficr(y_true, y_pred, capacity):
    valid = y_true >= capacity * VALID_RATIO_THRESHOLD
    if valid.sum() == 0:
        return "ficr", 0.0, True
    error_rate = np.abs(y_pred[valid] - y_true[valid]) / capacity
    unit_price = np.select([error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0)
    ficr = np.sum(y_true[valid] * unit_price) / np.sum(y_true[valid] * 4.0)
    return "ficr", ficr, True


def _true_ficr(y_true, y_pred, capacity):
    valid = y_true >= capacity * VALID_RATIO_THRESHOLD
    if valid.sum() == 0:
        return 0.0
    error_rate = np.abs(y_pred[valid] - y_true[valid]) / capacity
    unit_price = np.select([error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0)
    return np.sum(y_true[valid] * unit_price) / np.sum(y_true[valid] * 4.0)


# 1. 다항 선형 회귀
# ws117_cube_ldaps/gfs 등 이미 만들어둔 3승 피처가 X 안에 포함돼 있어서,
# 선형회귀만 태워도 풍속-발전량의 3차 곡선 관계를 반영하게 됨
def _fit_poly_ridge(X_tr, y_tr, sw_tr, X_val, y_val, capacity, ridge_params):
    model = Ridge(**ridge_params)
    model.fit(X_tr, y_tr, sample_weight=sw_tr)
    val_pred = model.predict(X_val)
    val_ficr = _true_ficr(y_val.values, val_pred, capacity)
    return model, val_ficr

# 2. MLP
class FICR_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(0.1)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# 대회 공식 FICR을 시그모이드로 부드럽게 근사한 손실함수 (미분 가능)
# unit_price(e) = 4 - sigmoid(k*(e-0.06)) - 3*sigmoid(k*(e-0.08))
# -> e<<0.06: 4, 0.06<e<0.08: 3, e>>0.08: 0 으로 부드럽게 이어짐 (계단이 아니라 S자 곡선)
def differentiable_ficr_loss(y_pred, y_true, capacity, k=200.0, sample_weight=None):
    diff = y_pred - y_true
    e = torch.abs(diff) / capacity
    s1 = torch.sigmoid(k * (e - 0.06))
    s2 = torch.sigmoid(k * (e - 0.08))
    unit_price = 4.0 - s1 - 3.0 * s2
    valid_mask = (y_true >= capacity * VALID_RATIO_THRESHOLD).float()
    if sample_weight is not None:
        valid_mask = valid_mask * sample_weight
    numerator = torch.sum(y_true * unit_price * valid_mask)
    denominator = torch.sum(y_true * 4.0 * valid_mask) + 1e-8
    return -(numerator / denominator)  # FICR 최대화 = -FICR 최소화


def _fit_mlp(X_tr, y_tr, sw_tr, X_val, y_val, capacity, mlp_params):
    torch.manual_seed(mlp_params["seed"])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32)
    sw_tr_t = torch.tensor(sw_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)

    model = FICR_MLP(X_tr.shape[1], mlp_params["hidden_dims"])
    optimizer = torch.optim.Adam(model.parameters(), lr=mlp_params["lr"], weight_decay=1e-5)

    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t, sw_tr_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=mlp_params["batch_size"], shuffle=True)

    best_val_ficr = -np.inf
    best_state = None
    patience_left = mlp_params["patience"]

    for epoch in range(mlp_params["max_epochs"]):
        model.train()
        for xb, yb, wb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            if epoch < mlp_params["warmup_epochs"]:
                loss = torch.mean(wb * (pred - yb) ** 2)
            else:
                loss = differentiable_ficr_loss(pred, yb, capacity, k=mlp_params["k"], sample_weight=wb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).numpy()
        val_ficr = _true_ficr(y_val.values, val_pred, capacity)  # 실제(계단식) FICR로 조기종료 판단

        if val_ficr > best_val_ficr:
            best_val_ficr = val_ficr
            best_state = {k_: v_.clone() for k_, v_ in model.state_dict().items()}
            patience_left = mlp_params["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model, scaler, best_val_ficr


def _predict_mlp(model, scaler, X):
    model.eval()
    X_s = scaler.transform(X)
    with torch.no_grad():
        pred = model(torch.tensor(X_s, dtype=torch.float32)).numpy()
    return pred

# 3. GAM
# spline_cols에 지정된 물리적 풍속/발전량 변수만 곡선(s)으로, 나머지는 선형(l)으로 적합
def _fit_gam(X_tr, y_tr, sw_tr, X_val, y_val, capacity, spline_cols, gam_params):
    feature_names = list(X_tr.columns)
    terms = None
    for i, col in enumerate(feature_names):
        term = s(i, n_splines=gam_params["n_splines"], lam=gam_params["lam"]) if col in spline_cols else l(i)
        terms = term if terms is None else terms + term

    model = LinearGAM(terms)
    model.fit(X_tr.values, y_tr.values, weights=sw_tr)
    val_pred = model.predict(X_val.values)
    val_ficr = _true_ficr(y_val.values, val_pred, capacity)
    return model, val_ficr

# ============================================
# Step 3: 손실함수 3종 (기존과 동일, 유지)
# ============================================
def make_huber_capacity_objective(capacity, delta, hess_floor=HUBER_HESS_FLOOR):
    delta_kwh = delta * capacity
    def _obj(y_true, y_pred, sample_weight=None):
        diff = y_pred - y_true
        denom = np.sqrt(1 + (diff / delta_kwh) ** 2)
        grad = diff / denom
        hess = np.maximum(1 / denom ** 3, hess_floor)
        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
        return grad, hess
    return _obj


# 수정 — error_rate만 남기지 않고 모델별 예측값 자체를 wide로 보관 (재블렌딩 가능하게)
def run_seasonal_wind_regime_cv(target, capacity, fold_starts, window_days=CV_WINDOW_DAYS,
                                 model_names=("xgb", "lgbm", "curve")):
    cfg = BEST_LOSS_CONFIG[target]
    lgbm_params = LGBM_PARAMS.copy()
    lgbm_params.update({'importance_type': 'gain'})

    all_rows = []
    for start in fold_starts:
        train_df, X_train, _, _, _ = get_tabular_data(mode="validation", validation_start=start)
        X_train_imp, _ = _prepare_common(train_df, X_train, X_train, train_df)

        X_tr, y_tr = get_target_xy(train_df, X_train_imp, target, subset="fit")
        X_v, y_v, ts_v = get_bounded_validation_xy(train_df, X_train_imp, target, start, window_days)
        if len(X_v) < 30:
            print(f"  [{target}] fold={start}: 검증 표본 부족({len(X_v)}) -> 스킵")
            continue

        tr_valid = y_tr >= capacity * VALID_RATIO_THRESHOLD
        if EXCLUDE_INVALID_ROWS:
            X_tr, y_tr = X_tr[tr_valid].reset_index(drop=True), y_tr[tr_valid].reset_index(drop=True)
            sw_tr = np.ones(len(y_tr))
        else:
            sw_tr = np.where(tr_valid, 1.0, INVALID_SAMPLE_WEIGHT)

        # [신규] 램프구간 샘플가중치 — 실제 배포 모델(_split_group_data)과 동일하게 반영
        ramp_range = RAMP_WS_RANGES.get(target)
        ws_col = WS_FEATURE_COL.get(target)
        if ramp_range is not None and ws_col is not None and ws_col in X_tr.columns:
            ws_tr = X_tr[ws_col].to_numpy()
            ramp_mask = (ws_tr >= ramp_range[0]) & (ws_tr < ramp_range[1])
            sw_tr = np.where(ramp_mask, sw_tr * RAMP_SAMPLE_WEIGHT, sw_tr)

        fold_result = pd.DataFrame({
            "fold": start,
            "ws": X_v[WS_FEATURE_COL[target]].to_numpy(),
            "y_true": y_v.to_numpy(),
        })

        if "xgb" in model_names:
            xgb_cfg = cfg["XGB"]
            best_xgb, _ = _fit_xgb_with_params(xgb_cfg["loss_name"], xgb_cfg["params"], X_tr, y_tr, sw_tr, X_v, y_v, capacity)
            fold_result["pred_xgb"] = best_xgb.predict(X_v)

        if "lgbm" in model_names:
            lgb_cfg = cfg["LGBM"]
            best_lgb, _ = _fit_lgbm_with_params(lgb_cfg["loss_name"], lgb_cfg["params"], lgbm_params, X_tr, y_tr, sw_tr, X_v, y_v, capacity)
            fold_result["pred_lgbm"] = best_lgb.predict(X_v)

        if "curve" in model_names:
            curve_model, _ = _fit_poly_ridge(X_tr, y_tr, sw_tr, X_v, y_v, capacity, RIDGE_PARAMS)
            fold_result["pred_curve"] = curve_model.predict(X_v)

        all_rows.append(fold_result)
        print(f"  [{target}] fold={start} 완료 (검증 {len(X_v)}행)")

    if not all_rows:
        raise ValueError(f"[{target}] 유효한 fold가 하나도 없습니다.")
    return pd.concat(all_rows, ignore_index=True)


def summarize_regime_cv(cv_wide, capacity, ws_bin_width=1.0, min_count=100):
    """wide-format cv 결과에서 풍속 bin별 모델 성능/최우수 모델을 집계한다 (진단용)."""
    model_cols = [c for c in cv_wide.columns if c.startswith("pred_")]
    valid = cv_wide["y_true"] >= capacity * VALID_RATIO_THRESHOLD
    rows = []
    for col in model_cols:
        name = col.replace("pred_", "")
        sub = cv_wide[valid].copy()
        sub["error_rate"] = (sub[col] - sub["y_true"]).abs() / capacity
        sub["ws_bin"] = np.floor(sub["ws"] / ws_bin_width) * ws_bin_width
        agg = sub.groupby("ws_bin").agg(n=("error_rate", "size"), mean_error_rate=("error_rate", "mean"))
        agg["model"] = name
        rows.append(agg.reset_index())
    comp = pd.concat(rows, ignore_index=True)
    pivot = comp.pivot(index="ws_bin", columns="model", values="mean_error_rate")
    n_pivot = comp.pivot(index="ws_bin", columns="model", values="n")
    pivot = pivot[n_pivot.min(axis=1) >= min_count]
    return pivot, pivot.idxmin(axis=1)


def make_threshold_weighted_objective(capacity, amplitude, sigma):
    grad_clip = capacity * 0.15                    # 약 3000~3200 kWh
    hess_clip = 1.0 + amplitude * 2.0 + 1.0         # amplitude가 커도 boundary 가중치가 안 잘리게

    def _obj(y_true, y_pred, sample_weight=None):
        diff = y_pred - y_true
        e = np.abs(diff) / capacity
        w = 1 + amplitude * (
            np.exp(-((e - 0.06) ** 2) / (2 * sigma ** 2)) +
            np.exp(-((e - 0.08) ** 2) / (2 * sigma ** 2))
        )
        grad = diff * w
        hess = np.ones_like(diff) * w

        grad = np.clip(grad, -grad_clip, grad_clip)
        hess = np.clip(hess, 1e-3, hess_clip)

        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
        return grad, hess
    return _obj

def make_smooth_ficr_objective(capacity, anchor_weight, k, scale_factor=1000.0):
    def _obj(y_true, y_pred, sample_weight=None):
        diff = y_pred - y_true
        e = np.abs(diff) / capacity
        
        # Sigmoid 곡선 (6%, 8% 경계)
        s1 = 1 / (1 + np.exp(-k * (e - 0.06)))
        s2 = 1 / (1 + np.exp(-k * (e - 0.08)))

        # 1차, 2차 미분
        dUP_de = -k * s1 * (1 - s1) - k * s2 * (1 - s2)
        d2UP_de2 = (-k ** 2 * s1 * (1 - s1) * (1 - 2 * s1)) + (-k ** 2 * s2 * (1 - s2) * (1 - 2 * s2))

        sign_diff = np.sign(diff)
        coef = (y_true / capacity) / 4.0

        dLoss_de = -coef * dUP_de
        d2Loss_de2 = -coef * d2UP_de2
        
        grad_ficr = dLoss_de * sign_diff / capacity
        hess_ficr_raw = d2Loss_de2 / (capacity ** 2)
        
        # [수정 1] 비볼록성(음수 헤시안) 해결 - 절댓값 처리
        hess_ficr_raw = np.abs(hess_ficr_raw)

        # [수정 2] Anchor 스케일 조정 - diff 단위가 아닌 비율 단위로 축소
        grad_anchor = diff / capacity
        hess_anchor = np.ones_like(diff) / capacity

        # 블렌딩
        grad_raw = anchor_weight * grad_anchor + (1 - anchor_weight) * grad_ficr
        hess_raw = anchor_weight * hess_anchor + (1 - anchor_weight) * hess_ficr_raw
        
        # [수정 3] 트리의 정규화(lambda)에 묻히지 않도록 전체 스케일 증폭
        grad = grad_raw * scale_factor
        hess = np.maximum(hess_raw * scale_factor, 1e-4)

        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
            
        return grad, hess
    return _obj


def make_huber_threshold_objective(capacity, delta, amplitude, sigma):
    """Huber의 스케일-안전한 그래디언트 위에, 6%/8% 경계 근처에서만
    국소적으로 그래디언트를 증폭시키는 하이브리드 손실.
    threshold_weighted처럼 raw diff를 그대로 곱하지 않고 배율(1+bump)로 얹으므로
    클리핑에 의존하지 않고도 스케일이 안전하게 유지된다."""
    delta_kwh = delta * capacity

    def _obj(y_true, y_pred, sample_weight=None):
        diff = y_pred - y_true
        e = np.abs(diff) / capacity

        denom = np.sqrt(1 + (diff / delta_kwh) ** 2)
        grad_huber = diff / denom
        hess_huber = 1 / denom ** 3

        boundary_bump = amplitude * (
            np.exp(-((e - 0.06) ** 2) / (2 * sigma ** 2)) +
            np.exp(-((e - 0.08) ** 2) / (2 * sigma ** 2))
        )
        grad = grad_huber * (1 + boundary_bump)
        hess = np.maximum(hess_huber * (1 + boundary_bump), HUBER_HESS_FLOOR)

        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
        return grad, hess
    return _obj


def make_threshold_weighted_huber(capacity, delta, amplitude, sigma, scale_factor=100.0):
    """
    전역적으로는 Huber(L1/L2 하이브리드)의 안정적인 기울기를 제공하여 이상치에 강건하게 대응하고,
    6%와 8% 오차 경계선 근처에서는 가중치(w)를 증폭시켜 모델이 해당 구간을 집중 타격하도록 유도합니다.
    """
    delta_kwh = delta * capacity
    def _obj(y_true, y_pred, sample_weight=None):
        diff = y_pred - y_true
        e = np.abs(diff) / capacity
        
        # 1. Huber Base: 전역적으로 안정적인 그래디언트 제공 (큰 오차 방치 금지)
        denom = np.sqrt(1 + (diff / delta_kwh) ** 2)
        grad_base = diff / denom
        hess_base = np.maximum(1 / denom ** 3, 1e-4)
        
        # 2. Boundary Weight: 6%와 8% 경계에서 가중치(자력) 증폭
        w = 1.0 + amplitude * (
            np.exp(-((e - 0.06) ** 2) / (2 * sigma ** 2)) +
            np.exp(-((e - 0.08) ** 2) / (2 * sigma ** 2))
        )
        
        # 3. 결합 및 스케일링: 작은 그래디언트/헤시안이 트리 모델의 lambda에 묻히지 않도록 스케일업
        grad = grad_base * w * scale_factor
        hess = hess_base * w * scale_factor
        
        # 방어 로직
        hess = np.maximum(hess, 1e-3)
        
        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
        return grad, hess
    return _obj

def make_asymmetric_zero_avoidance_objective(capacity, delta, amplitude, sigma, asymmetric_penalty=4.0, scale_factor=100.0):
    """
    과대예측(예측값 > 실제값)으로 인해 6% 및 8% 경계선을 넘어 0원 구간으로 
    추락할 위험이 있는 샘플에 대해 비대칭적으로 강력한 페널티를 부여하는 손실함수.
    """
    delta_kwh = delta * capacity
    def _obj(y_true, y_pred, sample_weight=None):
        diff = y_pred - y_true
        e = np.abs(diff) / capacity
        
        # 1. Huber Base: 전역적인 강건성(Robustness) 유지
        denom = np.sqrt(1 + (diff / delta_kwh) ** 2)
        grad_base = diff / denom
        hess_base = np.maximum(1 / denom ** 3, 1e-4)
        
        # 2. 기본 대칭 경계 가중치 (6%와 8% 부근)
        w = 1.0 + amplitude * (
            np.exp(-((e - 0.06) ** 2) / (2 * sigma ** 2)) +
            np.exp(-((e - 0.08) ** 2) / (2 * sigma ** 2))
        )
        
        # 3. 비대칭 패널티 (과대예측일 경우에만 가중치를 대폭 증폭)
        # diff > 0 이면 예측값이 실제값보다 큰 '과대예측' 상태
        is_overprediction = (diff > 0) & (e > 0.06)
        w = np.where(is_overprediction, w * asymmetric_penalty, w)
        
        # 4. 결합 및 스케일링
        grad = grad_base * w * scale_factor
        hess = hess_base * w * scale_factor
        
        hess = np.maximum(hess, 1e-3)
        
        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
        return grad, hess
    return _obj

LOSS_BUILDERS = {
    "huber_capacity": make_huber_capacity_objective,
    "threshold_weighted": make_threshold_weighted_objective,
    "smooth_ficr": make_smooth_ficr_objective,
    "threshold_weighted_huber": make_threshold_weighted_huber,
    "asymmetric_zero_avoidance": make_asymmetric_zero_avoidance_objective,
    "huber_threshold": make_huber_threshold_objective,
}


def _lgb_objective_wrapper(base_obj_fn, sample_weight):
    def _wrapped(y_true, y_pred):
        return base_obj_fn(y_true, y_pred, sample_weight=sample_weight)
    return _wrapped


class _OffsetModel:
    """예측을 항상 절대값(kWh) 스케일로 돌려주는 얇은 래퍼.
    내부 모델은 (y - offset)로 학습됐으므로, predict() 호출 시 offset을 자동으로 다시 더한다.
    train.py 어디서 .predict()를 호출하든 별도 처리 없이 항상 올바른 스케일이 나오도록 보장한다."""
    def __init__(self, model, offset):
        self.model = model
        self.offset = offset

    def predict(self, X):
        return self.model.predict(X) + self.offset

    def __getattr__(self, name):
        return getattr(self.model, name)  # feature_importances_ 등은 원본 모델에 위임

def _fit_xgb_with_params(loss_name, params, X_tr, y_tr, sw_tr, X_val, y_val, capacity, model_params=None):
    y_offset = float(y_tr.mean())
    y_tr_c = y_tr - y_offset
    y_val_c = y_val - y_offset

    base_obj_fn = LOSS_BUILDERS[loss_name](capacity, **params)
    # 클로저를 이용해 sw_tr 캡처
    def obj_fn(y_true, y_pred, sample_weight=None):
        return base_obj_fn(y_true, y_pred, sample_weight=sample_weight)
    xgb_kwargs = dict(model_params) if model_params is not None else {k: v for k, v in XGB_PARAMS.items()}
    model = XGBRegressor(
        **xgb_kwargs,
        objective=obj_fn,
        eval_metric=lambda y_true, y_pred: _neg_ficr(y_true + y_offset, y_pred + y_offset, capacity),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    model.fit(X_tr, y_tr_c, sample_weight=sw_tr, eval_set=[(X_val, y_val_c)], verbose=False)

    wrapped = _OffsetModel(model, y_offset)
    val_ficr = _true_ficr(y_val.values, wrapped.predict(X_val), capacity)
    return wrapped, val_ficr


def _fit_lgbm_with_params(loss_name, params, lgbm_params, X_tr, y_tr, sw_tr, X_val, y_val, capacity):
    y_offset = float(y_tr.mean())
    y_tr_c = y_tr - y_offset
    y_val_c = y_val - y_offset

    base_obj_fn = LOSS_BUILDERS[loss_name](capacity, **params)
    # 수정된 래퍼에 sw_tr 전달
    obj_fn = _lgb_objective_wrapper(base_obj_fn, sw_tr)
    model = LGBMRegressor(**lgbm_params, objective=obj_fn)
    model.fit(
        X_tr, y_tr_c, sample_weight=sw_tr,
        eval_set=[(X_val, y_val_c)],
        eval_metric=lambda y_true, y_pred: _lgb_ficr(y_true + y_offset, y_pred + y_offset, capacity),
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    wrapped = _OffsetModel(model, y_offset)
    val_ficr = _true_ficr(y_val.values, wrapped.predict(X_val), capacity)
    return wrapped, val_ficr


# ============================================
# 분위수 기반 전략적 예측 (GEFCom2014 winner 구조)
# ============================================
def _fit_quantile_lgbm(tau, X_tr, y_tr, sw_tr, X_val, y_val, lgbm_params):
    """단일 분위수(tau)에 대한 LightGBM pinball loss 모델을 학습한다.
    pinball loss는 오차 크기와 무관하게 그래디언트가 항상 tau 또는 tau-1이라
    smooth_ficr에서 겪었던 그래디언트 소실/부호반전 문제가 구조적으로 없다."""

    model = LGBMRegressor(**lgbm_params, objective="quantile", alpha=tau)
    model.fit(
        X_tr, y_tr, sample_weight=sw_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    print(f"    LGBM best_iteration={model.best_iteration_} / {lgbm_params['n_estimators']}")
    return model

def _fit_quantile_xgb(tau, X_tr, y_tr, sw_tr, X_val, y_val, xgb_params):
    """단일 분위수(tau)에 대한 XGBoost pinball loss 모델을 학습한다."""
    model = XGBRegressor(
        **xgb_params, 
        objective='reg:quantileerror', 
        quantile_alpha=tau, 
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )
    model.fit(
        X_tr, y_tr, sample_weight=sw_tr, 
        eval_set=[(X_val, y_val)], 
        verbose=False
    )
    print(f"    XGB best_iteration={model.best_iteration} / {xgb_params['n_estimators']}")
    return model


def _enforce_monotone_quantiles(pred_matrix, taus):
    """분위수 역전(crossing)을 단조증가 재배열(rearrangement)로 정리한다.
    Chernozhukov et al. (2010)의 rearrangement 기법 — tau 오름차순으로 정렬하고
    누적 최댓값을 취하면 단조성이 보장된다. GEFCom2014 준우승팀의 isotonic 후처리와 동일한 목적."""
    order = np.argsort(taus)
    sorted_preds = pred_matrix[:, order]
    monotone = np.maximum.accumulate(sorted_preds, axis=1)
    inverse_order = np.argsort(order)
    return monotone[:, inverse_order]

def diagnose_quantile_calibration(val_matrix, y_val, taus, target, capacity=None):
    order = np.argsort(taus)
    q = val_matrix[:, order]
    taus_sorted = np.array(sorted(taus))
    y = y_val.values if hasattr(y_val, "values") else np.asarray(y_val)

    if capacity is not None:
        valid = y >= capacity * VALID_RATIO_THRESHOLD
        excluded = (~valid).sum()
        y = y[valid]
        q = q[valid]
        print(f"\n  [{target}] 분위수 보정(calibration) 진단 (무효구간 {excluded}행 제외, {len(y)}행 기준)")
    else:
        print(f"\n  [{target}] 분위수 보정(calibration) 진단 (전체 {len(y)}행, 무효구간 미제외 — capacity 인자 없음)")

    print(f"  {'tau':>6} | {'실제 coverage':>12} | {'괴리':>8}")
    for i, tau in enumerate(taus_sorted):
        coverage = (y <= q[:, i]).mean()
        gap = coverage - tau
        flag = "  <-- 괴리 큼" if abs(gap) > 0.05 else ""
        print(f"  {tau:>6.2f} | {coverage:>12.3f} | {gap:>+8.3f}{flag}")


def _quantile_atom_weights(taus):
    """각 분위수 예측값에 부여할 확률질량(중점 기반 폭)을 계산한다. 합이 1이 되도록 정규화."""
    taus = np.asarray(sorted(taus), dtype=float)
    edges = np.concatenate([[0.0], (taus[:-1] + taus[1:]) / 2, [1.0]])
    weights = np.diff(edges)
    return weights  # taus 오름차순 기준


def _optimal_point_from_quantiles(pred_matrix, taus, capacity,
                                   margin_ratio=QUANTILE_SEARCH_MARGIN_RATIO,
                                   n_candidates=QUANTILE_SEARCH_N_CANDIDATES):
    order = np.argsort(taus)
    q = pred_matrix[:, order]  # (n_rows, K), tau 오름차순 정렬된 상태
    weights = _quantile_atom_weights(taus)  # (K,)

    # 극단적인 꼬리(0~20%, 80~100%)로 예측값이 도망가는 것을 방지
    idx_30 = max(0, int(len(taus) * 0.3))
    idx_70 = min(len(taus) - 1, int(len(taus) * 0.7))
    
    q_lo = q[:, idx_30:idx_30+1]
    q_hi = q[:, idx_70:idx_70+1]
    
    lo = np.clip(q_lo, 0, capacity)
    hi = np.clip(q_hi, 0, capacity)

    n_rows = q.shape[0]
    n_taus = q.shape[1]
    best_value = np.full(n_rows, -np.inf)
    best_point = q[:, len(taus) // 2].copy()  # 기본값: 중앙 분위수(대략 median)

    # 1. Base Candidates (단순 n등분)
    base_steps = np.linspace(0.0, 1.0, n_candidates)
    base_candidates = lo + base_steps[None, :] * (hi - lo)  # (n_rows, n_candidates)

    # 2. Strategic Candidates (각 분위수 값 기준 6%, 8% 경계선 바로 안쪽 핀포인트 타겟팅)
    shifts = np.array([0, -0.059, 0.059, -0.079, 0.079]) * capacity
    strategic_candidates = (q[:, :, None] + shifts).reshape(n_rows, -1)  # (n_rows, n_taus * 5)
    
    # 3. 모든 후보군 통합 및 클리핑
    all_candidates = np.concatenate([base_candidates, strategic_candidates], axis=1)
    all_candidates = np.clip(all_candidates, 0, capacity)

    # 탐색 루프 (배열의 열을 순회)
    for i in range(all_candidates.shape[1]):
        candidate = all_candidates[:, i]
        error_rate = np.abs(candidate[:, None] - q) / capacity
        unit_price = np.select(
            [error_rate <= 0.06, error_rate <= 0.08],
            [4.0, 3.0],
            default=0.0,
        )
        expected_value = (unit_price * weights[None, :]).sum(axis=1)

        improved = expected_value > best_value
        best_value = np.where(improved, expected_value, best_value)
        best_point = np.where(improved, candidate, best_point)

    return np.clip(best_point, 0, capacity)

def fit_quantile_calibration_offsets(calib_matrix, y_calib, taus, capacity=None):
    """보정용 셋에서 tau별 잔차의 tau-분위수를 구해 offset으로 반환한다.
    q_calibrated(x) = q_raw(x) + offset[tau] 형태로 적용하면
    P(y <= q_calibrated) ≈ tau가 되도록 보정된다 (conformal 보정과 동일한 원리)."""
    order = np.argsort(taus)
    taus_sorted = np.array(sorted(taus))
    q = calib_matrix[:, order]
    y = np.asarray(y_calib, dtype=float)

    if capacity is not None:
        valid = y >= capacity * VALID_RATIO_THRESHOLD
        y = y[valid]
        q = q[valid]

    offsets = np.zeros(len(taus_sorted))
    for i, tau in enumerate(taus_sorted):
        residual = y - q[:, i]
        offsets[i] = np.quantile(residual, tau)
    return offsets  # taus 오름차순 기준


def fit_quantile_calibration_offsets_seasonal(calib_matrix, y_calib, taus, calib_months, target_clusters, capacity=None):
    """월별 군집(시즌)을 기준으로 데이터를 나누어 각각의 특화된 Offset을 학습합니다."""
    # 1. 안전망: 표본 부족을 대비해 전체 데이터 기반 글로벌 Offset 사전 계산
    global_offsets = fit_quantile_calibration_offsets(calib_matrix, y_calib, taus, capacity)

    # 2. 월 -> 시즌 이름 맵핑 딕셔너리 생성
    month_to_season = {}
    for s_name, m_list in target_clusters.items():
        for m in m_list:
            month_to_season[m] = s_name

    # 3. 시즌별 Offset 계산
    season_offsets = {}
    for s_name in target_clusters.keys():
        # 해당 시즌에 속하는 행만 마스킹
        mask = np.array([month_to_season.get(m) == s_name for m in calib_months])
        
        # 보정 셋에 해당 시즌 표본이 50개 미만이면 오버피팅 방지를 위해 글로벌 Offset으로 대체
        if mask.sum() < 50:
            season_offsets[s_name] = global_offsets
        else:
            season_offsets[s_name] = fit_quantile_calibration_offsets(
                calib_matrix[mask], np.asarray(y_calib)[mask], taus, capacity
            )

    return season_offsets, month_to_season, global_offsets


def apply_quantile_calibration_seasonal(pred_matrix, taus, season_offsets, month_to_season, global_offsets, target_months):
    """예측 대상 월(Month)에 맞는 시즌별 Offset을 동적으로 적용합니다."""
    q_calibrated = pred_matrix.copy()
    order = np.argsort(taus)
    inverse_order = np.argsort(order)

    for i, m in enumerate(target_months):
        s_name = month_to_season.get(m)
        # 해당 월의 시즌 Offset을 가져오되, 매핑이 꼬이면 글로벌 Offset 사용
        offsets = season_offsets.get(s_name, global_offsets)
        
        # 보정 수행
        q_sorted = pred_matrix[i, order] + offsets
        q_calibrated[i] = q_sorted[inverse_order]

    return q_calibrated


def apply_quantile_calibration(pred_matrix, taus, offsets):
    """학습된 offset을 예측 분위수에 더해 보정한다."""
    order = np.argsort(taus)
    inverse_order = np.argsort(order)
    q = pred_matrix[:, order] + offsets[None, :]
    return q[:, inverse_order]


def train_quantile_ensemble(train_df, X_train, X_test, sample_sub, quantiles=QUANTILE_LEVELS):
    """그룹별로 여러 분위수 모델을 학습하고, 기대정산금 최대화 지점을 최종 예측으로 낸다.
    Purged K-Fold 기반의 OOF(Out-of-Fold) 예측을 통해 시계열 누수 없이 안전하게
    시즌별 분위수 보정(Calibration)을 수행한다."""
    print("--- [Mode: Quantile-Ensemble] GEFCom2014 winner 구조 (분위수 GBM + 기대정산금 최대화) ---")

    lgbm_params = LGBM_QUANTILE_PARAMS.copy()
    X_train_imp, X_test_imp = _prepare_common(train_df, X_train, X_test, sample_sub)

    ensemble_preds = pd.DataFrame(index=sample_sub.index)

    # 1. 클러스터(시즌) 정보 로드
    try:
        with open(MODEL_DIR / "seasonal_clusters.json", "r", encoding="utf-8") as f:
            seasonal_clusters = json.load(f)
    except FileNotFoundError:
        seasonal_clusters = {}
        print("[경고] seasonal_clusters.json 파일이 없습니다. 글로벌 보정으로 동작합니다.")

    for target in TARGET_COLS:
        capacity = CAPACITY_KWH[target]
        X_train_use, y_train_use, sw_use, X_tr, y_tr, sw_tr, X_val_final, y_val_final = _split_group_data(
            train_df, X_train_imp, target, capacity
        )
        
        # [신규] Purged K-Fold 설정 (시계열 쪼개기)
        from sklearn.model_selection import TimeSeriesSplit
        n_splits = 4
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # OOF(Out-of-Fold) 예측을 모을 그릇 준비
        oof_lgb_matrix = np.zeros((len(y_tr), len(quantiles)))
        oof_xgb_matrix = np.zeros((len(y_tr), len(quantiles)))
        
        # 테스트셋 예측을 누적할 그릇 (최종적으로 평균 냄)
        test_lgb_matrix = np.zeros((len(X_test_imp), len(quantiles)))
        test_xgb_matrix = np.zeros((len(X_test_imp), len(quantiles)))
        
        ws_col = WS_FEATURE_COL.get(target)
        mono_constraints_xgb = np.zeros(X_tr.shape[1], dtype=int)
        if ws_col in X_tr.columns:
            ws_idx = X_tr.columns.get_loc(ws_col)
            mono_constraints_xgb[ws_idx] = 1
            
        current_lgbm_params = LGBM_QUANTILE_PARAMS.copy()
        current_xgb_params = XGB_QUANTILE_PARAMS.copy()
        current_xgb_params["monotone_constraints"] = tuple(mono_constraints_xgb.tolist())

        print(f"\n  [{target}] Purged {n_splits}-Fold OOF 예측 및 모델 학습 시작...")
        
        # 2. K-Fold 순회하며 모델 학습 및 OOF 예측
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_tr)):
            print(f"    Fold {fold+1}/{n_splits} | 학습 {len(tr_idx)}행, 검증 {len(val_idx)}행")
            
            X_fold_tr = X_tr.iloc[tr_idx].reset_index(drop=True)
            y_fold_tr = y_tr.iloc[tr_idx].reset_index(drop=True)
            sw_fold_tr = sw_tr[tr_idx]
            
            X_fold_val = X_tr.iloc[val_idx].reset_index(drop=True)
            y_fold_val = y_tr.iloc[val_idx].reset_index(drop=True)
            
            for i, tau in enumerate(quantiles):
                # Fold별 모델 학습
                model_lgb = _fit_quantile_lgbm(tau, X_fold_tr, y_fold_tr, sw_fold_tr, X_fold_val, y_fold_val, current_lgbm_params)
                model_xgb = _fit_quantile_xgb(tau, X_fold_tr, y_fold_tr, sw_fold_tr, X_fold_val, y_fold_val, current_xgb_params)
                
                # 검증셋(OOF) 예측 채워넣기
                oof_lgb_matrix[val_idx, i] = model_lgb.predict(X_fold_val)
                oof_xgb_matrix[val_idx, i] = model_xgb.predict(X_fold_val)
                
                # 테스트셋 예측 누적 (나중에 1/n_splits)
                test_lgb_matrix[:, i] += model_lgb.predict(X_test_imp) / n_splits
                test_xgb_matrix[:, i] += model_xgb.predict(X_test_imp) / n_splits

        # 3. OOF 기반 앙상블 블렌딩 (가장 안전하고 검증된 예측값)
        oof_matrix = (oof_lgb_matrix + oof_xgb_matrix) / 2.0
        test_matrix = (test_lgb_matrix + test_xgb_matrix) / 2.0
        
        # 교차 금지(Monotonicity) 강제
        oof_matrix = _enforce_monotone_quantiles(oof_matrix, quantiles)
        test_matrix = _enforce_monotone_quantiles(test_matrix, quantiles)

        # 4. 시즌별 동적 Calibration (OOF 전체 데이터를 활용하므로 표본 부족 없음)
        calib_months = train_df.loc[X_tr.index, "forecast_kst_dtm"].dt.month.values
        test_months = pd.to_datetime(sample_sub["forecast_kst_dtm"]).dt.month.values
        target_clusters = seasonal_clusters.get(target, {})

        diagnose_quantile_calibration(oof_matrix, y_tr, quantiles, f"{target} (OOF blended, 보정 전)", capacity=capacity)

        if target_clusters:
            # 전체 3년 치 OOF 예측값을 바탕으로 시즌별 Offset 학습 (안전성 100%)
            season_offsets, month_to_season, global_offsets = fit_quantile_calibration_offsets_seasonal(
                oof_matrix, y_tr.values, quantiles, calib_months, target_clusters, capacity=capacity
            )
            
            # OOF와 테스트셋에 시즌별 보정 적용
            oof_matrix_cal = apply_quantile_calibration_seasonal(
                oof_matrix, quantiles, season_offsets, month_to_season, global_offsets, calib_months
            )
            test_matrix_cal = apply_quantile_calibration_seasonal(
                test_matrix, quantiles, season_offsets, month_to_season, global_offsets, test_months
            )
            
            oof_matrix_cal = _enforce_monotone_quantiles(oof_matrix_cal, quantiles)
            test_matrix_cal = _enforce_monotone_quantiles(test_matrix_cal, quantiles)
            print(f"  [{target}] 시즌별 동적 Calibration 완료 (OOF 기반)")
        else:
            offsets = fit_quantile_calibration_offsets(oof_matrix, y_tr.values, quantiles, capacity=capacity)
            oof_matrix_cal = _enforce_monotone_quantiles(apply_quantile_calibration(oof_matrix, quantiles, offsets), quantiles)
            test_matrix_cal = _enforce_monotone_quantiles(apply_quantile_calibration(test_matrix, quantiles, offsets), quantiles)

        diagnose_quantile_calibration(oof_matrix_cal, y_tr, quantiles, f"{target} (OOF blended, 보정 후)", capacity=capacity)

        # 5. 최종 기대정산금 최대화 지점 도출
        oof_point_raw = _optimal_point_from_quantiles(oof_matrix, quantiles, capacity)
        oof_point_cal = _optimal_point_from_quantiles(oof_matrix_cal, quantiles, capacity)
        ficr_raw = _true_ficr(y_tr.values, oof_point_raw, capacity)
        ficr_cal = _true_ficr(y_tr.values, oof_point_cal, capacity)

        print(f"  [{target}] [비교] 보정 전={ficr_raw:.4f} | 보정 후={ficr_cal:.4f}")

        if ficr_cal >= ficr_raw:
            final_test_matrix = test_matrix_cal
            print(f"  [{target}] -> 보정된 quantile 채택")
        else:
            final_test_matrix = test_matrix
            print(f"  [{target}] -> 보정이 손해, 원본 quantile 유지")

        test_point = _optimal_point_from_quantiles(final_test_matrix, quantiles, capacity)
        ensemble_preds[target] = np.clip(test_point, 0, capacity)

    return ensemble_preds



# ============================================
# Optuna 탐색 관련 (기존과 동일, 전부 유지 - 삭제하지 않음)
# ============================================
def _suggest_params(trial, loss_name):
    space = OPTUNA_SEARCH_SPACE[loss_name]
    params = {}
    for name, spec in space.items():
        if isinstance(spec["low"], int) and isinstance(spec["high"], int) and not spec.get("log", False):
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec.get("log", False) and isinstance(spec["low"], int):
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=True)
        else:
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return params

def _suggest_model_params(trial, model_type):
    """모델 구조 하이퍼파라미터(n_estimators, depth 등)를 제안하고,
    탐색 대상이 아닌 고정 설정(random_state, n_jobs 등)과 합쳐서 완전한 파라미터 dict를 반환."""
    if model_type == "XGB":
        space = XGB_SEARCH_SPACE
        base = {k: v for k, v in XGB_PARAMS.items() if k not in space}
    else:
        space = LGBM_SEARCH_SPACE
        base = {k: v for k, v in LGBM_PARAMS.items() if k not in space}
        base["importance_type"] = "gain"  # 기존 코드에서 하던 설정 유지

    suggested = {}
    for name, spec in space.items():
        if spec.get("type") == "int":
            suggested[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        else:
            suggested[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return {**base, **suggested}

def _run_optuna_search(model_type, target, capacity, lgbm_params, X_tr, y_tr, sw_tr, X_val, y_val, search_log,
                        warm_start=None):
    """손실함수별로 별도 study를 돌려 예산을 나누지 않음.
    warm_start: {"huber_capacity": {...}, "threshold_weighted": {...}, "smooth_ficr": {...}} 형태로
    각 손실함수별 시작점을 넣을 수 있음 (그룹별로 다르게 넣어야 효과적)."""
    best_overall = {"val_ficr": -np.inf, "loss_name": None, "params": None}

    trial_budget = {"huber_capacity": 20, "threshold_weighted": 40, "smooth_ficr": 40}

    for loss_name, n_trials_this_loss in trial_budget.items():
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
        )

        if warm_start and loss_name in warm_start:
            study.enqueue_trial(warm_start[loss_name])

        def _obj(trial, loss_name=loss_name):
            params = _suggest_params(trial, loss_name)  # 손실함수 파라미터
            model_params = _suggest_model_params(trial, model_type)  # [신규] 모델 구조 파라미터
            if model_type == "XGB":
                _, val_ficr = _fit_xgb_with_params(loss_name, params, X_tr, y_tr, sw_tr, X_val, y_val, capacity, model_params)
            else:
                _, val_ficr = _fit_lgbm_with_params(loss_name, params, model_params, X_tr, y_tr, sw_tr, X_val, y_val, capacity)
            trial.set_user_attr("params", params)
            trial.set_user_attr("model_params", model_params)  # [신규] 같이 기록
            return val_ficr

        study.optimize(_obj, n_trials=n_trials_this_loss, show_progress_bar=False)

        for t in study.trials:
            search_log.append({
                "target": target, "model": model_type, "loss_name": loss_name,
                "params": str(t.user_attrs.get("params")),
                "model_params": str(t.user_attrs.get("model_params")),
                "fold_ficrs": str(t.user_attrs.get("fold_ficrs")),
                "mean_val_ficr": t.value,
            })

        # [신규] 이 loss_name 안에서 "최고 단일 trial"이 아니라 mean-std 기준으로 대표 trial 선정
        best_trial_this_loss, best_score_this_loss = None, -np.inf
        for t in study.trials:
            fold_ficrs = t.user_attrs.get("fold_ficrs")
            if not fold_ficrs:
                continue
            score = float(np.mean(fold_ficrs)) - STABILITY_LAMBDA * float(np.std(fold_ficrs))
            if score > best_score_this_loss:
                best_score_this_loss = score
                best_trial_this_loss = t

        if best_trial_this_loss is None:
            continue

        if best_score_this_loss > best_overall["val_ficr"]:
            best_overall = {
                "val_ficr": best_score_this_loss,
                "loss_name": loss_name,
                "params": best_trial_this_loss.user_attrs["params"],
                "model_params": best_trial_this_loss.user_attrs["model_params"],
            }

    print(f"  [{target}][{model_type}] Optuna 최종 선택: {best_overall['loss_name']} "
          f"params={best_overall['params']} val_ficr={best_overall['val_ficr']:.4f}")

    if model_type == "XGB":
        best_model, _ = _fit_xgb_with_params(best_overall["loss_name"], best_overall["params"], X_tr, y_tr, sw_tr, X_val, y_val, capacity, best_overall["model_params"])
    else:
        best_model, _ = _fit_lgbm_with_params(best_overall["loss_name"], best_overall["params"], best_overall["model_params"], X_tr, y_tr, sw_tr, X_val, y_val, capacity)

    return {"model": best_model, **best_overall}

def _prepare_seasonal_fold_data(target, capacity, fold_starts):
    """계절별 fold마다 (X_tr, y_tr, sw_tr, X_v, y_v)를 미리 만들어 캐싱한다.
    Optuna trial마다 반복 재사용하기 위해 fold 준비는 탐색 시작 전 한 번만 수행한다."""
    fold_data = []
    for start in fold_starts:
        train_df, X_train, _, _, _ = get_tabular_data(mode="validation", validation_start=start)
        X_train_imp, _ = _prepare_common(train_df, X_train, X_train, train_df)

        X_tr, y_tr = get_target_xy(train_df, X_train_imp, target, subset="fit")
        X_v, y_v, _ = get_bounded_validation_xy(train_df, X_train_imp, target, start, CV_WINDOW_DAYS)
        if len(X_v) < 30:
            print(f"  [{target}] fold={start}: 검증 표본 부족({len(X_v)}) -> 스킵")
            continue

        tr_valid = y_tr >= capacity * VALID_RATIO_THRESHOLD
        if EXCLUDE_INVALID_ROWS:
            X_tr, y_tr = X_tr[tr_valid].reset_index(drop=True), y_tr[tr_valid].reset_index(drop=True)
            sw_tr = np.ones(len(y_tr))
        else:
            sw_tr = np.where(tr_valid, 1.0, INVALID_SAMPLE_WEIGHT)

        # 항목 3(샘플가중치)도 다중 fold 탐색에 동일하게 반영
        ramp_range = RAMP_WS_RANGES.get(target)
        ws_col = WS_FEATURE_COL.get(target)
        if ramp_range is not None and ws_col is not None and ws_col in X_tr.columns:
            ws_tr = X_tr[ws_col].to_numpy()
            ramp_mask = (ws_tr >= ramp_range[0]) & (ws_tr < ramp_range[1])
            sw_tr = np.where(ramp_mask, sw_tr * RAMP_SAMPLE_WEIGHT, sw_tr)

        fold_data.append({"fold": start, "X_tr": X_tr, "y_tr": y_tr, "sw_tr": sw_tr, "X_v": X_v, "y_v": y_v})
        print(f"  [{target}] fold={start} 준비 완료 (학습 {len(y_tr)}행, 검증 {len(y_v)}행)")

    if not fold_data:
        raise ValueError(f"[{target}] 유효한 fold가 하나도 없습니다.")
    return fold_data


def run_seasonal_optuna_search(model_type, target, capacity, lgbm_params, fold_data, search_log, warm_start=None):
    """단일 홀드아웃 대신 여러 계절 fold의 평균 val_ficr로 손실함수를 탐색한다."""
    best_overall = {"val_ficr": -np.inf, "loss_name": None, "params": None, "model_params": None}

    for loss_name, n_trials_this_loss in SEASONAL_OPTUNA_TRIAL_BUDGET.items():
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
        )
        if warm_start and loss_name in warm_start:
            study.enqueue_trial(warm_start[loss_name])

        def _obj(trial, loss_name=loss_name):
            params = _suggest_params(trial, loss_name)
            model_params = _suggest_model_params(trial, model_type)

            fold_ficrs = []
            for fold in fold_data:
                if model_type == "XGB":
                    _, val_ficr = _fit_xgb_with_params(
                        loss_name, params, fold["X_tr"], fold["y_tr"], fold["sw_tr"],
                        fold["X_v"], fold["y_v"], capacity, model_params,
                    )
                else:
                    _, val_ficr = _fit_lgbm_with_params(
                        loss_name, params, model_params, fold["X_tr"], fold["y_tr"], fold["sw_tr"],
                        fold["X_v"], fold["y_v"], capacity,
                    )
                fold_ficrs.append(val_ficr)
            if max(fold_ficrs) < 0.01:
                print(f"    [경고] {target}/{model_type}/{loss_name} trial: 전 fold FICR이 0에 가까움 "
                    f"(params={params}) -> 그래디언트 포화/학습 실패 의심")

            trial.set_user_attr("params", params)
            trial.set_user_attr("model_params", model_params)
            trial.set_user_attr("fold_ficrs", fold_ficrs)
            return float(np.mean(fold_ficrs))

        study.optimize(_obj, n_trials=n_trials_this_loss, show_progress_bar=False)

        for t in study.trials:
            search_log.append({
                "target": target, "model": model_type, "loss_name": loss_name,
                "params": str(t.user_attrs.get("params")),
                "model_params": str(t.user_attrs.get("model_params")),
                "fold_ficrs": str(t.user_attrs.get("fold_ficrs")),
                "mean_val_ficr": t.value,
            })

        best_trial_this_loss, best_score_this_loss = None, -np.inf
        for t in study.trials:
            fold_ficrs = t.user_attrs.get("fold_ficrs")
            if not fold_ficrs:
                continue
            score = float(np.mean(fold_ficrs)) - STABILITY_LAMBDA * float(np.std(fold_ficrs))
            if score > best_score_this_loss:
                best_score_this_loss, best_trial_this_loss = score, t

        if best_trial_this_loss is None:
            continue

        if best_score_this_loss > best_overall["val_ficr"]:
            best_overall = {
                "val_ficr": best_score_this_loss,
                "loss_name": loss_name,
                "params": best_trial_this_loss.user_attrs["params"],
                "model_params": best_trial_this_loss.user_attrs["model_params"],
            }

    print(f"  [{target}][{model_type}] 다중fold Optuna 최종 선택: {best_overall['loss_name']} "
          f"params={best_overall['params']} mean_val_ficr={best_overall['val_ficr']:.4f}")
    return best_overall


def _record_importance(raw_df, norm_df, target, name, imp_vals):
    col_name = f"{target}_{name}"
    raw_df[col_name] = imp_vals
    val_min, val_max = imp_vals.min(), imp_vals.max()
    norm_df[col_name] = (imp_vals - val_min) / (val_max - val_min) if val_max > val_min else np.zeros_like(imp_vals)

def _optimize_ensemble_weights(val_preds, y_val, capacity, seed=42):
    """검증셋에서 FICR을 최대화하는 모델별 앙상블 가중치(합=1, 0 이상)를 탐색.
    val_preds: [rf_val_pred, et_val_pred, xgb_val_pred, lgb_val_pred] 형태의 리스트
    FICR이 6%/8% 경계에서 끊기는 비연속 함수라 그래디언트 기반 최적화 대신
    블랙박스 전역 탐색(differential_evolution)을 사용."""
    val_preds = np.array(val_preds)  # shape: (n_models, n_samples)
    n_models = val_preds.shape[0]

    def neg_ficr_for_weights(raw_weights):
        w = np.clip(raw_weights, 0, None)
        if w.sum() == 0:
            return 0.0
        w = w / w.sum()
        blended = np.average(val_preds, axis=0, weights=w)
        return -_true_ficr(y_val, blended, capacity)

    result = differential_evolution(
        neg_ficr_for_weights, bounds=[(0, 1)] * n_models,
        seed=seed, maxiter=100, tol=1e-6, polish=True,
    )
    w = np.clip(result.x, 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(n_models) / n_models
    return w


def _prepare_common(train_df, X_train, X_test, sample_sub):
    """Optuna/하드코딩 두 함수가 공통으로 쓰는 전처리 (imputer, 결측/무효구간 처리)"""
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    return X_train_imp, X_test_imp


def _split_group_data(train_df, X_train_imp, target, capacity):
    # 1. 누수 방지 마스크 적용 (prepare_data.py의 get_target_xy 활용)
    X_tr, y_tr = get_target_xy(train_df, X_train_imp, target, subset="fit")
    X_val, y_val = get_target_xy(train_df, X_train_imp, target, subset="validation")
    X_train_use, y_train_use = get_target_xy(train_df, X_train_imp, target, subset="all")
    
    # 2. 검증 마스크가 지정되지 않은 경우(예: data_mode="final") 기존처럼 Holdout 비율로 분할
    if len(X_val) == 0:
        split_idx = int(len(y_tr) * (1 - VAL_HOLDOUT_RATIO))
        X_val = X_tr.iloc[split_idx:].reset_index(drop=True)
        y_val = y_tr.iloc[split_idx:].reset_index(drop=True)
        X_tr = X_tr.iloc[:split_idx].reset_index(drop=True)
        y_tr = y_tr.iloc[:split_idx].reset_index(drop=True)

    # 3. 무효구간(10% 미만) 가중치/제외 처리 (Train 셋)
    tr_valid = y_tr >= capacity * VALID_RATIO_THRESHOLD
    if EXCLUDE_INVALID_ROWS:
        X_tr = X_tr[tr_valid].reset_index(drop=True)
        y_tr = y_tr[tr_valid].reset_index(drop=True)
        sw_tr = np.ones(len(y_tr))
    else:
        sw_tr = np.where(tr_valid, 1.0, INVALID_SAMPLE_WEIGHT)

    # 풍속구간별 샘플가중치: 램프구간 표본에 가중치 상향
    ramp_range = RAMP_WS_RANGES.get(target)
    ws_col = WS_FEATURE_COL.get(target)
    if ramp_range is not None and ws_col is not None and ws_col in X_tr.columns:
        ws_tr = X_tr[ws_col].to_numpy()
        ramp_mask = (ws_tr >= ramp_range[0]) & (ws_tr < ramp_range[1])
        sw_tr = np.where(ramp_mask, sw_tr * RAMP_SAMPLE_WEIGHT, sw_tr)
    HIGH_OUTPUT_THRESHOLD = 0.70
    HIGH_OUTPUT_WEIGHT = 2.5
    
    high_output_mask = y_tr >= (capacity * HIGH_OUTPUT_THRESHOLD)
    sw_tr = np.where(high_output_mask, sw_tr * HIGH_OUTPUT_WEIGHT, sw_tr)
    
    # 전체 Train셋(sw_use)에도 동일하게 적용
    if 'sw_use' in locals():
        high_output_mask_use = y_train_use >= (capacity * HIGH_OUTPUT_THRESHOLD)
        sw_use = np.where(high_output_mask_use, sw_use * HIGH_OUTPUT_WEIGHT, sw_use)

    # 4. 무효구간 처리 (Train 전체 셋)
    use_valid = y_train_use >= capacity * VALID_RATIO_THRESHOLD
    if EXCLUDE_INVALID_ROWS:
        X_train_use = X_train_use[use_valid].reset_index(drop=True)
        y_train_use = y_train_use[use_valid].reset_index(drop=True)
        sw_use = np.ones(len(y_train_use))
    else:
        sw_use = np.where(use_valid, 1.0, INVALID_SAMPLE_WEIGHT)

    # Train 전체 셋(sw_use)에도 램프구간 샘플가중치 동일 적용
    if ramp_range is not None and ws_col is not None and ws_col in X_train_use.columns:
        ws_use = X_train_use[ws_col].to_numpy()
        ramp_mask_use = (ws_use >= ramp_range[0]) & (ws_use < ramp_range[1])
        sw_use = np.where(ramp_mask_use, sw_use * RAMP_SAMPLE_WEIGHT, sw_use)

    print(f"  [{target}] 학습 {len(y_tr)}행, hyperparam-val {len(y_val)}행 "
          f"(무효구간 처리: {'제외' if EXCLUDE_INVALID_ROWS else f'가중치 {INVALID_SAMPLE_WEIGHT}'})")

    return X_train_use, y_train_use, sw_use, X_tr, y_tr, sw_tr, X_val, y_val


# ============================================
# 기존: Optuna 탐색 기반 학습 (그대로 유지, 삭제하지 않음)
# ============================================
def train_ensemble(train_df, X_train, X_test, sample_sub):
    print("--- [Mode: Ensemble-Search] Training with Optuna Search ---")

    lgbm_params = LGBM_PARAMS.copy()
    lgbm_params.update({'importance_type': 'gain'})

    X_train_imp, X_test_imp = _prepare_common(train_df, X_train, X_test, sample_sub)

    ensemble_preds = pd.DataFrame(index=sample_sub.index)
    raw_feature_importances = pd.DataFrame(index=X_train_imp.columns)
    norm_feature_importances = pd.DataFrame(index=X_train_imp.columns)
    search_log = []
    final_selection_log = []

    for target in TARGET_COLS:
        capacity = CAPACITY_KWH[target]
        X_train_use, y_train_use, sw_use, X_tr, y_tr, sw_tr, X_val, y_val = _split_group_data(
            train_df, X_train_imp, target, capacity
        )

        preds = []

        # rf = RandomForestRegressor(**RF_PARAMS)
        # rf.fit(X_train_use, y_train_use, sample_weight=sw_use)
        # preds.append(rf.predict(X_test_imp))
        # print(f"  [{target}] RF Trained.")
        # _record_importance(raw_feature_importances, norm_feature_importances, target, "RF", rf.feature_importances_)

        # et = ExtraTreesRegressor(**ET_PARAMS)
        # et.fit(X_train_use, y_train_use, sample_weight=sw_use)
        # preds.append(et.predict(X_test_imp))
        # print(f"  [{target}] ET Trained.")
        # _record_importance(raw_feature_importances, norm_feature_importances, target, "ET", et.feature_importances_)

        # 그룹별 워밍업 값 (직전 그리드서치/Optuna 최고 기록을 시작점으로)
        warm_start = BEST_LOSS_CONFIG.get(target, {})
        xgb_warm = {warm_start.get("XGB", {}).get("loss_name"): warm_start.get("XGB", {}).get("params")} if "XGB" in warm_start else None
        lgb_warm = {warm_start.get("LGBM", {}).get("loss_name"): warm_start.get("LGBM", {}).get("params")} if "LGBM" in warm_start else None

        print(f"  [{target}] XGB - Optuna 탐색 중...")
        best_xgb = _run_optuna_search("XGB", target, capacity, lgbm_params, X_tr, y_tr, sw_tr, X_val, y_val, search_log, warm_start=xgb_warm)
        preds.append(best_xgb["model"].predict(X_test_imp))
        diagnose_ficr_boundary(
            y_val.values, best_xgb["model"].predict(X_val), capacity, target=target, model_name="XGB",
            wind_speed=X_val[WS_FEATURE_COL[target]] if WS_FEATURE_COL.get(target) in X_val.columns else None,
            save_dir=OUTPUT_DIR,
        )
        final_selection_log.append({"target": target, "model": "XGB", "loss_name": best_xgb["loss_name"],
                                     "params": str(best_xgb["params"]), "val_ficr": best_xgb["val_ficr"]})
        _record_importance(raw_feature_importances, norm_feature_importances, target, "XGB", best_xgb["model"].feature_importances_)

        print(f"  [{target}] LGBM - Optuna 탐색 중...")
        best_lgb = _run_optuna_search("LGBM", target, capacity, lgbm_params, X_tr, y_tr, sw_tr, X_val, y_val, search_log, warm_start=lgb_warm)
        preds.append(best_lgb["model"].predict(X_test_imp))
        diagnose_ficr_boundary(
            y_val.values, best_lgb["model"].predict(X_val), capacity, target=target, model_name="LGBM",
            wind_speed=X_val[WS_FEATURE_COL[target]] if WS_FEATURE_COL.get(target) in X_val.columns else None,
            save_dir=OUTPUT_DIR,
        )
        final_selection_log.append({"target": target, "model": "LGBM", "loss_name": best_lgb["loss_name"],
                                     "params": str(best_lgb["params"]), "val_ficr": best_lgb["val_ficr"]})
        imp_vals = best_lgb["model"].feature_importances_
        if imp_vals.sum() > 0:
            imp_vals = imp_vals / imp_vals.sum()
        _record_importance(raw_feature_importances, norm_feature_importances, target, "LGBM", imp_vals)

        # ensemble_preds[target] = np.clip(np.mean(preds, axis=0), 0, capacity)  # [기존] RF/ET/XGB/LGBM 단순 평균
        print(f"  [{target}] Ensemble: XGB+LGBM only (RF/ET excluded)")
        ensemble_preds[target] = np.clip(np.mean(preds, axis=0), 0, capacity)

    raw_feature_importances['mean_importance_norm'] = norm_feature_importances.mean(axis=1)
    sorted_fi = raw_feature_importances.sort_values(by='mean_importance_norm', ascending=False)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fi_path = OUTPUT_DIR / f"feature_importance_calibrated_{timestamp_str}.csv"
    sorted_fi.to_csv(fi_path, encoding='utf-8-sig')
    print(f"\n📊 보정된 변수 중요도 저장 완료: {fi_path}")

    search_log_df = pd.DataFrame(search_log)
    search_log_path = OUTPUT_DIR / f"optuna_search_full_{timestamp_str}.csv"
    search_log_df.to_csv(search_log_path, index=False, encoding='utf-8-sig')
    print(f"📊 Optuna 전체 시행 로그 저장 완료: {search_log_path} (총 {len(search_log_df)}행)")

    final_log_df = pd.DataFrame(final_selection_log)
    final_log_path = OUTPUT_DIR / f"optuna_final_selection_{timestamp_str}.csv"
    final_log_df.to_csv(final_log_path, index=False, encoding='utf-8-sig')
    print(f"📊 최종 선택 로그 저장 완료: {final_log_path}")
    print(final_log_df.to_string(index=False))

    return ensemble_preds


# ============================================
# 신규: 하드코딩된 최적 파라미터로 바로 학습 (탐색 없음, 빠름)
# ============================================
def train_ensemble_final(train_df, X_train, X_test, sample_sub):
    print("--- [Mode: Ensemble-Final] Training with Hardcoded Best Params (BEST_LOSS_CONFIG) ---")

    lgbm_params = LGBM_PARAMS.copy()
    lgbm_params.update({'importance_type': 'gain'})

    X_train_imp, X_test_imp = _prepare_common(train_df, X_train, X_test, sample_sub)

    try:
        regime_all_cfg = load_regime_config(MODEL_DIR / "regime_config_monthly.json")
        print(f"regime_config.json 로드 완료: {REGIME_CONFIG_PATH}")
    except FileNotFoundError:
        regime_all_cfg = {}
        print(f"[경고] {REGIME_CONFIG_PATH} 없음 -> 전 그룹 단순평균 폴백. scripts/discover_regimes.py 먼저 실행하세요.")

    ensemble_preds = pd.DataFrame(index=sample_sub.index)
    raw_feature_importances = pd.DataFrame(index=X_train_imp.columns)
    norm_feature_importances = pd.DataFrame(index=X_train_imp.columns)
    final_used_log = []

    for target in TARGET_COLS:
        capacity = CAPACITY_KWH[target]
        X_train_use, y_train_use, sw_use, X_tr, y_tr, sw_tr, X_val, y_val = _split_group_data(
            train_df, X_train_imp, target, capacity
        )

        # rf = RandomForestRegressor(**RF_PARAMS)
        # rf.fit(X_train_use, y_train_use, sample_weight=sw_use)
        # preds.append(rf.predict(X_test_imp))
        # val_preds.append(rf.predict(X_val))  # [신규]
        # print(f"  [{target}] RF Trained.")
        # _record_importance(raw_feature_importances, norm_feature_importances, target, "RF", rf.feature_importances_)

        # et = ExtraTreesRegressor(**ET_PARAMS)
        # et.fit(X_train_use, y_train_use, sample_weight=sw_use)
        # preds.append(et.predict(X_test_imp))
        # val_preds.append(et.predict(X_val))  # [신규]
        # print(f"  [{target}] ET Trained.")
        # _record_importance(raw_feature_importances, norm_feature_importances, target, "ET", et.feature_importances_)

        target_regime = regime_all_cfg.get(target)   
        use_curve = False
        if target_regime is not None:
            # 월별 중첩 구조이므로 모든 월의 regimes를 순회하며 확인
            for m_str, m_data in target_regime.items():
                if any(r["weights"].get("curve", 0) > 0 for r in m_data["regimes"]):
                    use_curve = True
                    break

        model_val_preds = {}
        model_test_preds = {}

        if use_curve:
            curve_model, curve_val_ficr = _fit_poly_ridge(X_tr, y_tr, sw_tr, X_val, y_val, capacity, RIDGE_PARAMS)
            model_val_preds["curve"] = curve_model.predict(X_val)
            model_test_preds["curve"] = curve_model.predict(X_test_imp)
            print(f"  [{target}] Ridge(curve) Trained. val_ficr={curve_val_ficr:.4f}")
            
        cfg = BEST_LOSS_CONFIG[target]
        model_cfg = BEST_MODEL_CONFIG.get(target, {})  # [신규]

        xgb_cfg = cfg["XGB"]
        xgb_model_params = model_cfg.get("XGB") or {"random_state": 42, "n_jobs": -1}
        lgb_cfg = cfg["LGBM"]
        lgb_model_params = {**lgbm_params, **model_cfg.get("LGBM", {})}

        if target == "kpx_group_3":
            # 사전 학습을 위한 Group 1, 2 데이터 병합
            X_src_list, y_src_list, sw_src_list = [], [], []
            for src_target in ["kpx_group_1", "kpx_group_2"]:
                X_src, y_src = get_target_xy(train_df, X_train_imp, src_target, subset="fit")
                src_valid = y_src >= CAPACITY_KWH[src_target] * VALID_RATIO_THRESHOLD
                sw_src = np.where(src_valid, 1.0, INVALID_SAMPLE_WEIGHT)
                X_src_list.append(X_src); y_src_list.append(y_src); sw_src_list.append(sw_src)
            
            X_source = pd.concat(X_src_list, ignore_index=True)
            y_source = pd.concat(y_src_list, ignore_index=True)
            sw_source = np.concatenate(sw_src_list)
            capacity_source = CAPACITY_KWH["kpx_group_1"]

            # [수정됨] Zero-shot 대신 Transfer Learning(사전학습 + 미세조정) 적용
            best_xgb, xgb_val_ficr = _fit_xgb_transfer(
                target, capacity, capacity_source, X_source, y_source, sw_source,
                X_tr, y_tr, sw_tr, X_val, y_val, 
                xgb_cfg["loss_name"], xgb_cfg["params"], xgb_model_params,
                finetune_round_ratio=0.4, finetune_early_stopping_rounds=150
            )
            
            best_lgb, lgb_val_ficr = _fit_lgbm_transfer(
                target, capacity, capacity_source, X_source, y_source, sw_source,
                X_tr, y_tr, sw_tr, X_val, y_val, 
                lgb_cfg["loss_name"], lgb_cfg["params"], lgb_model_params,
                finetune_round_ratio=0.4, finetune_early_stopping_rounds=150
            )
        else:
            best_xgb, xgb_val_ficr = _fit_xgb_with_params(
                xgb_cfg["loss_name"], xgb_cfg["params"], X_tr, y_tr, sw_tr, X_val, y_val, capacity, xgb_model_params
            )
            best_lgb, lgb_val_ficr = _fit_lgbm_with_params(
                lgb_cfg["loss_name"], lgb_cfg["params"], lgb_model_params, X_tr, y_tr, sw_tr, X_val, y_val, capacity
            )

        model_test_preds["xgb"] = best_xgb.predict(X_test_imp)
        model_val_preds["xgb"] = best_xgb.predict(X_val)
        print(f"  [{target}] XGB Trained. loss={xgb_cfg['loss_name']} params={xgb_cfg['params']} val_ficr={xgb_val_ficr:.4f}")
        diagnose_ficr_boundary(
            y_val.values, best_xgb.predict(X_val), capacity, target=target, model_name="XGB",
            wind_speed=X_val[WS_FEATURE_COL[target]] if WS_FEATURE_COL.get(target) in X_val.columns else None,
            save_dir=OUTPUT_DIR,
        )
        final_used_log.append({"target": target, "model": "XGB", "loss_name": xgb_cfg["loss_name"],
                                "params": str(xgb_cfg["params"]), "val_ficr": xgb_val_ficr})
        _record_importance(raw_feature_importances, norm_feature_importances, target, "XGB", best_xgb.feature_importances_)

        model_test_preds["lgbm"] = best_lgb.predict(X_test_imp)
        model_val_preds["lgbm"] = best_lgb.predict(X_val)
        print(f"  [{target}] LGBM Trained. loss={lgb_cfg['loss_name']} params={lgb_cfg['params']} val_ficr={lgb_val_ficr:.4f}")
        diagnose_ficr_boundary(
            y_val.values, best_lgb.predict(X_val), capacity, target=target, model_name="LGBM",
            wind_speed=X_val[WS_FEATURE_COL[target]] if WS_FEATURE_COL.get(target) in X_val.columns else None,
            save_dir=OUTPUT_DIR,
        )
        final_used_log.append({"target": target, "model": "LGBM", "loss_name": lgb_cfg["loss_name"],
                                "params": str(lgb_cfg["params"]), "val_ficr": lgb_val_ficr})
        imp_vals = best_lgb.feature_importances_
        if imp_vals.sum() > 0:
            imp_vals = imp_vals / imp_vals.sum()
        _record_importance(raw_feature_importances, norm_feature_importances, target, "LGBM", imp_vals)

        # ensemble_preds[target] = np.clip(np.mean(preds, axis=0), 0, capacity)
        # [신규] 검증셋 FICR 최대화 가중치로 앙상블
        # weights = _optimize_ensemble_weights(val_preds, y_val.values, capacity)
        # print(f"  [{target}] Ensemble weights (RF/ET/XGB/LGBM): {np.round(weights, 3)}")
        # weighted_pred = np.average(preds, axis=0, weights=weights)
        # ensemble_preds[target] = np.clip(weighted_pred, 0, capacity)
        # 수정 — regime_config.json 그대로 적용 + 모니터링용 로그만 남김(동작에 영향 없음)
        ws_col = WS_FEATURE_COL[target]

        if target_regime is not None:
            regime_val_pred = apply_regime_config(
                model_val_preds, X_val[ws_col], 
                train_df.loc[X_val.index, "forecast_kst_dtm"], # 시간 정보 추가
                target_regime, capacity, target
            )
            regime_val_ficr = _true_ficr(y_val.values, regime_val_pred, capacity)
            simple_avg_val_ficr = _true_ficr(y_val.values, np.mean(list(model_val_preds.values()), axis=0), capacity)
            print(f"  [{target}] [모니터링, hyperparam-val] regime_config={regime_val_ficr:.4f} "
                f"vs 단순평균={simple_avg_val_ficr:.4f}")

            if regime_val_ficr >= simple_avg_val_ficr:
                raw_pred = apply_regime_config(
                    model_test_preds, X_test_imp[ws_col], 
                    sample_sub["forecast_kst_dtm"], 
                    target_regime, capacity, target
                )
                print(f"  [{target}] -> regime_config 채택")
            else:
                raw_pred = np.clip(np.mean(list(model_test_preds.values()), axis=0), 0, capacity)
                print(f"  [{target}] [주의] regime_config가 단순평균보다 낮음 -> 단순평균으로 자동 폴백.")

            ensemble_preds[target] = np.clip(raw_pred, 0, capacity)
        else:
            print(f"  [{target}] regime_config 없음 -> 단순평균 폴백. scripts/discover_regimes.py를 먼저 실행하세요.")
            ensemble_preds[target] = np.clip(np.mean(list(model_test_preds.values()), axis=0), 0, capacity)

    raw_feature_importances['mean_importance_norm'] = norm_feature_importances.mean(axis=1)
    sorted_fi = raw_feature_importances.sort_values(by='mean_importance_norm', ascending=False)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fi_path = OUTPUT_DIR / f"feature_importance_final_{timestamp_str}.csv"
    sorted_fi.to_csv(fi_path, encoding='utf-8-sig')
    print(f"\n📊 변수 중요도 저장 완료: {fi_path}")

    final_log_df = pd.DataFrame(final_used_log)
    final_log_path = OUTPUT_DIR / f"final_config_used_{timestamp_str}.csv"
    final_log_df.to_csv(final_log_path, index=False, encoding='utf-8-sig')
    print(f"📊 사용된 최종 설정 저장 완료: {final_log_path}")
    print(final_log_df.to_string(index=False))

    return ensemble_preds

### 발전량 확인용
def diagnose_ficr_boundary(y_true, y_pred, capacity, target, model_name, wind_speed=None, save_dir=None):
    """
    검증셋 예측/실제값의 오차율을 6%/8% 경계선 기준으로 진단.
    - error_rate 히스토그램 (6%/8% 경계선 표시)
    - 가격 구간(4원/3원/0원)별 표본 수·비중
    - (wind_speed 제공 시) 풍속 구간별로 '경계선 바로 바깥(0원대)' 비중이 몰리는지 확인
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = y_true >= capacity * VALID_RATIO_THRESHOLD
    error_rate = np.abs(y_pred[valid] - y_true[valid]) / capacity

    # 1) 가격 구간별 집계
    tier = np.select(
        [error_rate <= 0.06, error_rate <= 0.08, error_rate > 0.08],
        ["4원 (~6%)", "3원 (6~8%)", "0원 (8%초과)"],
        default="Unknown"
    )
    tier_counts = pd.Series(tier).value_counts()
    tier_ratio = (tier_counts / len(tier) * 100).round(1)
    print(f"\n[{target}][{model_name}] Price tier distribution {valid.sum()}행 기준 가격 구간별 분포")
    print(pd.DataFrame({"count": tier_counts, "ratio(%)": tier_ratio}))

    # 2) '경계선 바로 바깥'에 몰려있는지 확인 (6~7%, 8~9% vs 그 외)
    near_6 = ((error_rate > 0.06) & (error_rate <= 0.07)).sum()
    near_8 = ((error_rate > 0.08) & (error_rate <= 0.09)).sum()
    total_over_6 = (error_rate > 0.06).sum()
    total_over_8 = (error_rate > 0.08).sum()
    print(f"  6% 초과 중 6~7%(경계선 바로 바깥) 비중: "
          f"{near_6}/{total_over_6} ({100*near_6/max(total_over_6,1):.1f}%)")
    print(f"  8% 초과 중 8~9%(경계선 바로 바깥) 비중: "
          f"{near_8}/{total_over_8} ({100*near_8/max(total_over_8,1):.1f}%)")

    # 3) 히스토그램 (6%/8% 경계선 표시)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(error_rate, bins=60, alpha=0.7)
    ax.axvline(0.06, color="red", linestyle="--", label="6% boundary")
    ax.axvline(0.08, color="orange", linestyle="--", label="8% boundary")
    ax.set_xlabel("error_rate = |pred-true| / capacity")
    ax.set_ylabel("count")
    ax.set_title(f"{target} - {model_name} Error Rate Distribution (validation)")
    ax.legend()
    if save_dir:
        fig.savefig(save_dir / f"ficr_boundary_{target}_{model_name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 4) (선택) 풍속 구간별 '0원대(8% 초과)' 비중 - 어느 풍속대에서 몰리는지 확인
    if wind_speed is not None:
        wind_speed = np.asarray(wind_speed)[valid]
        ws_bin = np.floor(wind_speed / 1.0)  # 1 m/s 단위로 뭉뚱그려서 확인
        df_ws = pd.DataFrame({"ws_bin": ws_bin, "error_rate": error_rate})
        df_ws["zero_price"] = df_ws["error_rate"] > 0.08
        by_ws = df_ws.groupby("ws_bin").agg(
            n=("error_rate", "size"),
            zero_price_ratio=("zero_price", "mean"),
        )
        by_ws = by_ws[by_ws["n"] >= 10]  # 표본 너무 적은 구간 제외
        print(f"\n  풍속(m/s) 구간별 0원 처리 비중 (표본 10개 이상만):")
        print(by_ws.sort_values("zero_price_ratio", ascending=False).head(10))

    return error_rate


def diagnose_quantile_calibration_seasonal(target, capacity, fold_starts, quantiles=QUANTILE_LEVELS):
    """discover_loss_config.py와 동일한 계절 fold 구조로 quantile 모델을 학습하고,
    fold 전체를 합쳐 coverage를 진단한다. 단일 홀드아웃(hyperparam-val)의 계절 편향을 배제한 결과."""
    fold_data = _prepare_seasonal_fold_data(target, capacity, fold_starts)

    all_y = []
    all_lgb_preds = {tau: [] for tau in quantiles}
    all_xgb_preds = {tau: [] for tau in quantiles}

    for fold in fold_data:
        X_tr, y_tr, sw_tr, X_v, y_v = fold["X_tr"], fold["y_tr"], fold["sw_tr"], fold["X_v"], fold["y_v"]

        ws_col = WS_FEATURE_COL.get(target)
        mono = np.zeros(X_tr.shape[1], dtype=int)
        if ws_col in X_tr.columns:
            mono[X_tr.columns.get_loc(ws_col)] = 1

        lgbm_params = LGBM_QUANTILE_PARAMS.copy()
        xgb_params = XGB_QUANTILE_PARAMS.copy()
        xgb_params["monotone_constraints"] = tuple(mono.tolist())

        for tau in quantiles:
            model_lgb = _fit_quantile_lgbm(tau, X_tr, y_tr, sw_tr, X_v, y_v, lgbm_params)
            model_xgb = _fit_quantile_xgb(tau, X_tr, y_tr, sw_tr, X_v, y_v, xgb_params)
            all_lgb_preds[tau].append(model_lgb.predict(X_v))
            all_xgb_preds[tau].append(model_xgb.predict(X_v))

        all_y.append(y_v.values if hasattr(y_v, "values") else y_v)
        print(f"  [{target}] fold={fold['fold']} 완료")

    y_all = np.concatenate(all_y)
    lgb_matrix = np.column_stack([np.concatenate(all_lgb_preds[tau]) for tau in quantiles])
    xgb_matrix = np.column_stack([np.concatenate(all_xgb_preds[tau]) for tau in quantiles])
    lgb_matrix = _enforce_monotone_quantiles(lgb_matrix, quantiles)
    xgb_matrix = _enforce_monotone_quantiles(xgb_matrix, quantiles)

    diagnose_quantile_calibration(lgb_matrix, y_all, quantiles, f"{target} (LGBM, 4-fold 합산)", capacity=capacity)
    diagnose_quantile_calibration(xgb_matrix, y_all, quantiles, f"{target} (XGB, 4-fold 합산)", capacity=capacity)


# 풍속 구간별 모델 비교
def compare_models_by_wind_bin(y_true, model_val_preds, wind_speed, capacity, target,
                                ws_bin_width=1.0, min_count=10):
    """
    같은 풍속 구간(1m/s 단위, 기존 diagnose_ficr_boundary와 동일 기준)에서
    모델별 error_rate / 0원비중을 나란히 비교한다.
    model_val_preds: {"xgb": pred, "lgbm": pred, "curve": pred}
    """
    y_true = np.asarray(y_true)
    wind_speed = np.asarray(wind_speed)
    valid = y_true >= capacity * VALID_RATIO_THRESHOLD
    ws_bin = np.floor(wind_speed[valid] / ws_bin_width) * ws_bin_width

    rows = []
    for name, pred in model_val_preds.items():
        pred = np.asarray(pred)[valid]
        error_rate = np.abs(pred - y_true[valid]) / capacity
        df = pd.DataFrame({"ws_bin": ws_bin, "error_rate": error_rate})
        df["zero_price"] = df["error_rate"] > 0.08
        agg = df.groupby("ws_bin").agg(
            n=("error_rate", "size"),
            mean_error_rate=("error_rate", "mean"),
            zero_price_ratio=("zero_price", "mean"),
        )
        agg = agg[agg["n"] >= min_count]
        agg["model"] = name
        rows.append(agg.reset_index())

    comp = pd.concat(rows, ignore_index=True)
    pivot_err = comp.pivot(index="ws_bin", columns="model", values="mean_error_rate")
    pivot_zp = comp.pivot(index="ws_bin", columns="model", values="zero_price_ratio")

    print(f"\n[{target}] 풍속 구간별 모델 비교 - 평균 error_rate")
    print(pivot_err.round(4))
    print(f"\n[{target}] 풍속 구간별 모델 비교 - 0원(8%초과) 비중")
    print(pivot_zp.round(3))

    best_model_by_bin = pivot_err.idxmin(axis=1)
    print(f"\n[{target}] 구간별 error_rate 최우수 모델:\n{best_model_by_bin}")
    return pivot_err, pivot_zp, best_model_by_bin

# 풍속 구간별 조건부 블렌딩
def load_regime_config(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for target, months_cfg in raw.items():
        for month_str, month_data in months_cfg.items():
            for r in month_data["regimes"]:
                r["lo"] = -np.inf if r["lo"] is None else r["lo"]
                r["hi"] = np.inf if r["hi"] is None else r["hi"]
    return raw


def apply_regime_config(model_test_preds, wind_speed, forecast_kst_dtm, regime_cfg_all, capacity, target):
    """계절별 군집(Seasonal Clusters) 단위로 매핑된 구간 가중치 규칙을 동적으로 적용한다."""
    ws = np.asarray(wind_speed)
    dt = pd.to_datetime(forecast_kst_dtm)
    months = np.asarray(dt.dt.month)
    out = np.zeros(len(ws))
    assigned = np.zeros(len(ws), dtype=bool)
    
    # 1. 월(Month) -> 시즌(Season) 매핑을 역으로 유추하기 위한 사전 준비 
    # (일반적으로 seasonal_clusters.json 구조에 맞춤)
    try:
        with open(MODEL_DIR / "seasonal_clusters.json", "r", encoding="utf-8") as f:
            seasonal_clusters = json.load(f)
    except FileNotFoundError:
        seasonal_clusters = {}

    for target_col, seasons_dict in regime_cfg_all.items():
        pass # regime_cfg_all은 이미 특정 그룹의 dict임
        
    # 만약 regime_cfg_all이 그룹별 설정이라면, 각 시즌에 속하는 월(Month) 리스트를 순회
    # seasonal_clusters 구조가 없거나 복잡할 경우를 대비해 월을 직접 시즌으로 맵핑
    group_clusters = seasonal_clusters.get(target, {})
    print(f"    [디버그] target={target} 사용 중인 클러스터: {group_clusters}")
    for season_name, season_data in regime_cfg_all.items():
        target_months = group_clusters.get(season_name, [])
        
        if not target_months:
            continue
            
        season_mask = np.isin(months, target_months)
        if not season_mask.any():
            continue
            
        regimes = season_data["regimes"]
        
        for r in regimes:
            mask = season_mask & (ws >= r["lo"]) & (ws < r["hi"]) & (~assigned)
            if not mask.any():
                continue
                
            w = r["weights"]
            names = [n for n in w if w[n] > 0 and n in model_test_preds]
            if not names:
                names = list(model_test_preds.keys())
                weights = np.ones(len(names)) / len(names)
            else:
                weights = np.array([w[n] for n in names])
                weights = weights / weights.sum()
                
            arr = np.array([model_test_preds[n][mask] for n in names])
            out[mask] = np.average(arr, axis=0, weights=weights)
            assigned |= mask
            
    if (~assigned).any():
        arr = np.array(list(model_test_preds.values()))
        out[~assigned] = np.mean(arr[:, ~assigned], axis=0)
        
    return np.clip(out, 0, capacity)

### 일반화 성능 ###
def _fit_xgb_transfer(target, capacity, capacity_source,
                       X_source, y_source, sw_source,
                       X_tr, y_tr, sw_tr, X_val, y_val,
                       loss_name, loss_params, model_params,
                       finetune_round_ratio=0.4,
                       finetune_early_stopping_rounds=150):
    y_source_scaled = y_source / capacity_source * capacity

    y_offset = float(y_tr.mean())
    y_source_c = y_source_scaled - y_offset
    y_tr_c = y_tr - y_offset
    y_val_c = y_val - y_offset

    base_obj_fn = LOSS_BUILDERS[loss_name](capacity, **loss_params)
    # XGBoost가 sample_weight를 자동으로 넣어줄 수 있도록, 반드시 이 시그니처 그대로 정의
    def obj_fn(y_true, y_pred, sample_weight=None):
        return base_obj_fn(y_true, y_pred, sample_weight=sample_weight)

    # 1) 사전학습 (source) — .fit(sample_weight=sw_source)를 주면 XGBoost가 obj_fn에 자동 전달
    pre_kwargs = dict(model_params)
    pre_model = XGBRegressor(**pre_kwargs, objective=obj_fn)
    pre_model.fit(X_source, y_source_c, sample_weight=sw_source, verbose=False)

    # 2) fine-tune (target) — xgb_model=로 이어서 학습
    finetune_kwargs = dict(model_params)
    finetune_kwargs["n_estimators"] = max(
        int(model_params.get("n_estimators", 300) * finetune_round_ratio),
        finetune_early_stopping_rounds + 20,   # patience보다 넉넉하게 cap을 잡음
    )
    finetuned = XGBRegressor(
        **finetune_kwargs,
        objective=obj_fn,   # 같은 obj_fn 재사용 — .fit(sample_weight=sw_tr)가 자동으로 들어감
        eval_metric=lambda yt, yp: _neg_ficr(yt + y_offset, yp + y_offset, capacity),
        early_stopping_rounds=finetune_early_stopping_rounds,
    )
    finetuned.fit(X_tr, y_tr_c, sample_weight=sw_tr, eval_set=[(X_val, y_val_c)],
                  xgb_model=pre_model.get_booster(), verbose=False)
    print(f"    [디버그] {target} XGB fine-tune early_stopping_rounds={finetune_early_stopping_rounds}, "
          f"best_iteration={finetuned.best_iteration} / cap={finetune_kwargs['n_estimators']}")

    wrapped = _OffsetModel(finetuned, y_offset)
    val_ficr = _true_ficr(y_val.values, wrapped.predict(X_val), capacity)
    return wrapped, val_ficr


def _fit_lgbm_transfer(target, capacity, capacity_source,
                        X_source, y_source, sw_source,
                        X_tr, y_tr, sw_tr, X_val, y_val,
                        loss_name, loss_params, lgbm_params,
                        finetune_round_ratio=0.4,
                        finetune_early_stopping_rounds=150):
    y_source_scaled = y_source / capacity_source * capacity
    y_offset = float(y_tr.mean())
    y_source_c = y_source_scaled - y_offset
    y_tr_c = y_tr - y_offset
    y_val_c = y_val - y_offset

    base_obj = LOSS_BUILDERS[loss_name](capacity, **loss_params)

    pre_model = LGBMRegressor(**lgbm_params, objective=_lgb_objective_wrapper(base_obj, sw_source))
    pre_model.fit(X_source, y_source_c, sample_weight=sw_source)

    finetune_params = dict(lgbm_params)
    finetune_params["n_estimators"] = max(
        int(lgbm_params.get("n_estimators", 300) * finetune_round_ratio),
        finetune_early_stopping_rounds + 20,
    )
    finetuned = LGBMRegressor(**finetune_params, objective=_lgb_objective_wrapper(base_obj, sw_tr))
    finetuned.fit(
        X_tr, y_tr_c, sample_weight=sw_tr,
        eval_set=[(X_val, y_val_c)],
        eval_metric=lambda yt, yp: _lgb_ficr(yt + y_offset, yp + y_offset, capacity),
        callbacks=[lgb.early_stopping(stopping_rounds=finetune_early_stopping_rounds, verbose=False)],
        init_model=pre_model.booster_,
    )
    print(f"    [디버그] {target} LGBM fine-tune early_stopping_rounds={finetune_early_stopping_rounds}, "
          f"best_iteration={finetuned.best_iteration_} / cap={finetune_params['n_estimators']}")

    wrapped = _OffsetModel(finetuned, y_offset)
    val_ficr = _true_ficr(y_val.values, wrapped.predict(X_val), capacity)
    return wrapped, val_ficr

def _fit_xgb_zeroshot(target, capacity, capacity_source,
                       X_source, y_source, sw_source,
                       X_val, y_val, loss_name, loss_params, model_params):
    """fine-tune 없이 group1+2 모델을 group3에 그대로 적용."""
    y_source_scaled = y_source / capacity_source * capacity
    y_offset = float(y_source_scaled.mean())
    y_source_c = y_source_scaled - y_offset

    base_obj_fn = LOSS_BUILDERS[loss_name](capacity, **loss_params)
    def obj_fn(y_true, y_pred, sample_weight=None):
        return base_obj_fn(y_true, y_pred, sample_weight=sample_weight)

    model = XGBRegressor(**dict(model_params), objective=obj_fn)
    model.fit(X_source, y_source_c, sample_weight=sw_source, verbose=False)

    wrapped = _OffsetModel(model, y_offset)
    val_ficr = _true_ficr(y_val.values, wrapped.predict(X_val), capacity)
    return wrapped, val_ficr

def _fit_lgbm_zeroshot(target, capacity, capacity_source,
                        X_source, y_source, sw_source,
                        X_val, y_val, loss_name, loss_params, lgbm_params):
    """fine-tune 없이 group1+2 모델을 group3에 그대로 적용 (LGBM)."""
    y_source_scaled = y_source / capacity_source * capacity
    y_offset = float(y_source_scaled.mean())
    y_source_c = y_source_scaled - y_offset

    base_obj = LOSS_BUILDERS[loss_name](capacity, **loss_params)

    model = LGBMRegressor(**lgbm_params, objective=_lgb_objective_wrapper(base_obj, sw_source))
    model.fit(X_source, y_source_c, sample_weight=sw_source)

    wrapped = _OffsetModel(model, y_offset)
    val_ficr = _true_ficr(y_val.values, wrapped.predict(X_val), capacity)
    return wrapped, val_ficr



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["search", "final"], default="final",
                         help="search: Optuna 재탐색 / final: 하드코딩된 최적값 + 월별 다이내믹 앙상블 학습 (기본값)")
    args = parser.parse_args()

    data_mode = "final"
    train_df, X_train, test_df, X_test, sample_sub = get_tabular_data(mode=data_mode)

    if args.mode == "search":
        final_preds = train_ensemble(train_df, X_train, X_test, sample_sub)
    else:
        # 주력 아키텍처: 최적 손실함수(XGB/LGBM) + 월별 다이내믹 앙상블(regime_config_monthly.json)
        final_preds = train_ensemble_final(train_df, X_train, X_test, sample_sub)

    submission = sample_sub[["forecast_id", "forecast_kst_dtm"]].copy()
    for col in TARGET_COLS:
        submission[col] = final_preds[col]

    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{args.mode}_{timestamp_str}.csv"
    output_path = OUTPUT_DIR / filename

    submission.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 파이프라인 완료! 저장 위치: {output_path}")


if __name__ == "__main__":
    main()