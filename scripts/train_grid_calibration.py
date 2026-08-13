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
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s, l
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


from config import (
    TARGET_COLS, CAPACITY_KWH, RF_PARAMS, ET_PARAMS, XGB_PARAMS, LGBM_PARAMS, OUTPUT_DIR,
    VALID_RATIO_THRESHOLD, EXCLUDE_INVALID_ROWS, INVALID_SAMPLE_WEIGHT,
    VAL_HOLDOUT_RATIO, EARLY_STOPPING_ROUNDS,
    OPTUNA_N_TRIALS, OPTUNA_SEED, OPTUNA_SEARCH_SPACE, HUBER_HESS_FLOOR,
    BEST_LOSS_CONFIG, XGB_SEARCH_SPACE, LGBM_SEARCH_SPACE, BEST_MODEL_CONFIG,
    RIDGE_PARAMS, MLP_PARAMS, GAM_SPLINE_COLS, GAM_PARAMS,
    REGIME_CONFIG_PATH, CV_WINDOW_DAYS,
    WS_FEATURE_COL, RAMP_WS_RANGES, RAMP_SAMPLE_WEIGHT, SEASONAL_OPTUNA_TRIAL_BUDGET,
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
    def _obj(y_true, y_pred, sample_weight=None):
        diff = y_pred - y_true
        e = np.abs(diff) / capacity
        w = 1 + amplitude * (
            np.exp(-((e - 0.06) ** 2) / (2 * sigma ** 2)) +
            np.exp(-((e - 0.08) ** 2) / (2 * sigma ** 2))
        )
        grad = diff * w
        hess = np.ones_like(diff) * w

        grad = np.clip(grad, -3.0, 3.0)
        hess = np.clip(hess, 1e-3, 5.0)
        
        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
        return grad, hess
    return _obj


def make_smooth_ficr_objective(capacity, anchor_weight, k):
    def _obj(y_true, y_pred, sample_weight=None):
        diff = y_pred - y_true
        e = np.abs(diff) / capacity
        s1 = 1 / (1 + np.exp(-k * (e - 0.06)))
        s2 = 1 / (1 + np.exp(-k * (e - 0.08)))

        dUP_de = -k * s1 * (1 - s1) - k * s2 * (1 - s2)
        d2UP_de2 = (-k ** 2 * s1 * (1 - s1) * (1 - 2 * s1)) + (-k ** 2 * s2 * (1 - s2) * (1 - 2 * s2))

        sign_diff = np.sign(diff)
        coef = (y_true / capacity) / 4.0

        dLoss_de = -coef * dUP_de
        d2Loss_de2 = -coef * d2UP_de2
        grad_ficr = dLoss_de * sign_diff
        hess_ficr_raw = d2Loss_de2

        grad_anchor = diff
        hess_anchor = np.ones_like(diff)

        grad = anchor_weight * grad_anchor + (1 - anchor_weight) * grad_ficr
        hess_raw = anchor_weight * hess_anchor + (1 - anchor_weight) * hess_ficr_raw
        hess = np.maximum(hess_raw, 1e-3)

        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
        return grad, hess
    return _obj


LOSS_BUILDERS = {
    "huber_capacity": make_huber_capacity_objective,
    "threshold_weighted": make_threshold_weighted_objective,
    "smooth_ficr": make_smooth_ficr_objective,
}


def _lgb_objective_wrapper(base_obj_fn):
    def _wrapped(y_true, y_pred):
        return base_obj_fn(y_true, y_pred, sample_weight=None)
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

    obj_fn = LOSS_BUILDERS[loss_name](capacity, **params)
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
    obj_fn = _lgb_objective_wrapper(base_obj_fn)
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
                "model_params": str(t.user_attrs.get("model_params")),  # [신규]
                "val_ficr": t.value,
            })

        if study.best_value > best_overall["val_ficr"]:
            best_overall = {
                "val_ficr": study.best_value,
                "loss_name": loss_name,
                "params": study.best_trial.user_attrs["params"],
                "model_params": study.best_trial.user_attrs["model_params"],  # [신규]
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

        if study.best_value > best_overall["val_ficr"]:
            best_overall = {
                "val_ficr": study.best_value,
                "loss_name": loss_name,
                "params": study.best_trial.user_attrs["params"],
                "model_params": study.best_trial.user_attrs["model_params"],
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

    # 4. 무효구간 처리 (Train 전체 셋)
    use_valid = y_train_use >= capacity * VALID_RATIO_THRESHOLD
    if EXCLUDE_INVALID_ROWS:
        X_train_use = X_train_use[use_valid].reset_index(drop=True)
        y_train_use = y_train_use[use_valid].reset_index(drop=True)
        sw_use = np.ones(len(y_train_use))
    else:
        sw_use = np.where(use_valid, 1.0, INVALID_SAMPLE_WEIGHT)

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
        regime_all_cfg = load_regime_config(REGIME_CONFIG_PATH)
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

        target_regime = regime_all_cfg.get(target)   # 함수 맨 앞에서 한 번 로드해둔 dict, 아래 참고
        use_curve = target_regime is not None and any(
            r["weights"].get("curve", 0) > 0 for r in target_regime["regimes"]
        )

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
        xgb_model_params = model_cfg.get("XGB") or None  # [신규] 비어있으면 None -> 기존 XGB_PARAMS 사용
        best_xgb, xgb_val_ficr = _fit_xgb_with_params(
            xgb_cfg["loss_name"], xgb_cfg["params"], X_tr, y_tr, sw_tr, X_val, y_val, capacity, xgb_model_params
        )
        model_test_preds["xgb"] = best_xgb.predict(X_test_imp)
        model_val_preds["xgb"] = best_xgb.predict(X_val)
        print(f"  [{target}] XGB Trained. loss={xgb_cfg['loss_name']} params={xgb_cfg['params']} val_ficr={xgb_val_ficr:.4f}")
        print(f"   [{target}] XGB best_iteration={best_xgb.best_iteration} / {XGB_PARAMS['n_estimators']}")
        diagnose_ficr_boundary(
            y_val.values, best_xgb.predict(X_val), capacity, target=target, model_name="XGB",
            wind_speed=X_val[WS_FEATURE_COL[target]] if WS_FEATURE_COL.get(target) in X_val.columns else None,
            save_dir=OUTPUT_DIR,
        )
        final_used_log.append({"target": target, "model": "XGB", "loss_name": xgb_cfg["loss_name"],
                                "params": str(xgb_cfg["params"]), "val_ficr": xgb_val_ficr})
        _record_importance(raw_feature_importances, norm_feature_importances, target, "XGB", best_xgb.feature_importances_)

        lgb_cfg = cfg["LGBM"]
        lgb_model_params = {**lgbm_params, **model_cfg.get("LGBM", {})}  # [신규] 있으면 덮어쓰기, 없으면 기존 lgbm_params 그대로
        best_lgb, lgb_val_ficr = _fit_lgbm_with_params(
            lgb_cfg["loss_name"], lgb_cfg["params"], lgb_model_params, X_tr, y_tr, sw_tr, X_val, y_val, capacity
        )
        model_test_preds["lgbm"] = best_lgb.predict(X_test_imp)
        model_val_preds["lgbm"] = best_lgb.predict(X_val)
        print(f"  [{target}] LGBM Trained. loss={lgb_cfg['loss_name']} params={lgb_cfg['params']} val_ficr={lgb_val_ficr:.4f}")
        print(f"   [{target}] LGBM best_iteration={best_lgb.best_iteration_} / {LGBM_PARAMS['n_estimators']}")
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
            regime_val_pred = apply_regime_config(model_val_preds, X_val[ws_col], target_regime, capacity)
            regime_val_ficr = _true_ficr(y_val.values, regime_val_pred, capacity)
            simple_avg_val_ficr = _true_ficr(y_val.values, np.mean(list(model_val_preds.values()), axis=0), capacity)
            print(f"  [{target}] [모니터링, hyperparam-val] regime_config={regime_val_ficr:.4f} "
                f"vs 단순평균={simple_avg_val_ficr:.4f}")

            if regime_val_ficr >= simple_avg_val_ficr:
                ensemble_preds[target] = apply_regime_config(model_test_preds, X_test_imp[ws_col], target_regime, capacity)
                print(f"  [{target}] -> regime_config 채택")
            else:
                ensemble_preds[target] = np.clip(np.mean(list(model_test_preds.values()), axis=0), 0, capacity)
                print(f"  [{target}] [주의] regime_config가 단순평균보다 낮음 -> 단순평균으로 자동 폴백. "
                    f"discover_regimes.py 재실행 권장")
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
    for target, cfg in raw.items():
        for r in cfg["regimes"]:
            r["lo"] = -np.inf if r["lo"] is None else r["lo"]
            r["hi"] = np.inf if r["hi"] is None else r["hi"]
    return raw


def apply_regime_config(model_test_preds, wind_speed, regime_cfg, capacity):
    """discover_regimes.py가 만든 그룹별 구간 규칙을 예측값에 적용한다."""
    ws = np.asarray(wind_speed)
    out = np.zeros(len(ws))
    assigned = np.zeros(len(ws), dtype=bool)
    for r in regime_cfg["regimes"]:
        mask = (ws >= r["lo"]) & (ws < r["hi"]) & (~assigned)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["search", "final"], default="final",
                         help="search: Optuna 재탐색 (느림, 시간 있을 때) / final: 하드코딩된 최적값으로 바로 학습 (빠름, 기본값)")
    args = parser.parse_args()

    # 데이터 호출 시 train.py의 모드(search)와 충돌하는 에러를 방지하기 위해 
    # 일반적인 예측 과정용인 "final" 모드를 고정 인자로 전달.
    data_mode = "final"
    train_df, X_train, test_df, X_test, sample_sub = get_tabular_data(mode=data_mode)

    if args.mode == "search":
        final_preds = train_ensemble(train_df, X_train, X_test, sample_sub)
    else:
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