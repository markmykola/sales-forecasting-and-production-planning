# core/forecasting.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from pmdarima import auto_arima
    _HAS_PMDARIMA = True
except Exception:
    auto_arima = None
    _HAS_PMDARIMA = False

from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

# ---------- 1. Завантаження даних ----------
def load_rossmann(train_path, store_path):
    train = pd.read_csv(train_path, parse_dates=["Date"])
    train["Date"] = train["Date"] + pd.DateOffset(years=9)
    store = pd.read_csv(store_path)
    df = pd.merge(train, store, on="Store", how="left")
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    df["StateHoliday"] = df["StateHoliday"].astype(str).replace("0", "0").fillna("0")
    for col in ["Sales", "Customers", "Promo", "SchoolHoliday", "Open"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["StateHoliday_Numeric"] = le.fit_transform(df["StateHoliday"].astype(str))
    return df

# ---------- 2. Підготовка даних для одного магазину ----------
def prepare_store_series(df, store_id):
    sub = df[df["Store"] == store_id].copy()
    if sub.empty:
        raise ValueError(f"No data for Store {store_id}")
    sub = sub.sort_values("Date")
    sub.set_index("Date", inplace=True)
    for lag in (1, 7, 30):
        sub[f"Sales_lag{lag}"] = sub["Sales"].shift(lag)
    sub["DayOfWeek"] = sub.index.dayofweek
    sub["Month"] = sub.index.month
    for c in ["Promo", "SchoolHoliday", "StateHoliday_Numeric", "Open"]:
        if c not in sub.columns:
            sub[c] = 0
    sub = sub.sort_index()
    sub.dropna(inplace=True)
    return sub

# ---------- helper ----------
def _safe_log_transform(series):
    eps = 1e-6
    return np.log(series + eps)

def _safe_exp_transform(series):
    return np.exp(series)

# ---------- SARIMA ----------
def sarima_forecast(series, horizon=30, use_log=True, future_exog_df=None):
    y = series["Sales"].copy()
    last_date = y.index[-1]
    if use_log:
        y_t = _safe_log_transform(y)
    else:
        y_t = y

    exog_cols = [c for c in ["Promo", "SchoolHoliday", "StateHoliday_Numeric", "Open"] if c in series.columns]
    exog = series[exog_cols] if exog_cols else None
    future_exog = None
    
    if exog is not None and future_exog_df is not None:
        future_exog = future_exog_df[exog_cols].iloc[:horizon]

    try:
        model = SARIMAX(y_t, exog=exog, 
                        order=(1, 1, 1), 
                        seasonal_order=(0, 1, 1, 7),
                        enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)
        fc_obj = fitted.get_forecast(steps=horizon, exog=future_exog)
        fc = fc_obj.predicted_mean
        
        if use_log:
            fc = _safe_exp_transform(fc)
            
        fc.index = pd.date_range(last_date + timedelta(days=1), periods=horizon, freq="D")
        fc.name = "forecast"
        return fc, fitted
    except Exception as e:
        print(f"SARIMA fallback error: {e}")
        idx = pd.date_range(last_date + timedelta(days=1), periods=horizon, freq="D")
        last = float(series["Sales"].iloc[-1])
        return pd.Series([last] * horizon, index=idx, name="forecast"), None

def sarima_conf_int(fitted, steps=30, alpha=0.05, future_exog_df=None):
    if fitted is None:
        return None
    try:
        future_exog = None
        if future_exog_df is not None and hasattr(fitted, 'model') and fitted.model.exog_names:
            exog_cols = fitted.model.exog_names
            future_exog = future_exog_df[exog_cols].iloc[:steps]

        forecast = fitted.get_forecast(steps=steps, exog=future_exog)
        ci = forecast.conf_int(alpha=alpha)
        last_date = fitted.data.dates[-1]
        ci.index = pd.date_range(last_date + timedelta(days=1), periods=steps, freq="D")
        return ci
    except Exception:
        return None

# ---------- AutoARIMA ----------
def auto_arima_forecast(series, horizon=30, future_exog_df=None):
    y = series["Sales"].copy()
    last_date = y.index[-1]
    if not _HAS_PMDARIMA:
        return sarima_forecast(series, horizon=horizon, future_exog_df=future_exog_df)

    exog_cols = [c for c in ["Promo", "SchoolHoliday", "StateHoliday_Numeric", "Open"] if c in series.columns]
    exog = series[exog_cols] if exog_cols else None

    try:
        if exog is not None and len(exog) >= 1:
            model = auto_arima(y, exogenous=exog, seasonal=True, m=7, suppress_warnings=True, stepwise=True)
            if future_exog_df is not None:
                future_exog = future_exog_df[exog_cols].iloc[:horizon]
            else:
                future_exog = exog.iloc[-horizon:].copy() 
            fc_vals = model.predict(n_periods=horizon, exogenous=future_exog)
        else:
            model = auto_arima(y, seasonal=True, m=7, suppress_warnings=True, stepwise=True)
            fc_vals = model.predict(n_periods=horizon)
            
        idx = pd.date_range(last_date + timedelta(days=1), periods=horizon, freq="D")
        return pd.Series(fc_vals, index=idx, name="forecast"), model
    except Exception:
        return sarima_forecast(series, horizon=horizon, future_exog_df=future_exog_df)

# ---------- Random Forest ----------
def rf_forecast(series, horizon=30, future_exog_df=None):
    df = series.copy()
    features = ["Sales_lag1", "Sales_lag7", "Sales_lag30", 
                "DayOfWeek", "Month", 
                "Promo", "SchoolHoliday", "StateHoliday_Numeric", "Open"]
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df["Sales"]

    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1) 
    model.fit(X, y)

    if future_exog_df is not None:
        future = future_exog_df.copy()
    else:
        future = df.iloc[-horizon:].copy() 

    preds = []
    for i in range(horizon):
        X_pred_row = future.iloc[i].copy() 
        X_pred_row["Sales_lag1"] = preds[-1] if preds else df["Sales"].iloc[-1]
        X_pred_row["Sales_lag7"] = preds[-7] if len(preds) >= 7 else df["Sales"].iloc[-(7 - len(preds))]
        X_pred_row["Sales_lag30"] = preds[-30] if len(preds) >= 30 else df["Sales"].iloc[-(30 - len(preds))]
        
        X_pred = X_pred_row[features].to_frame().T.values
        y_pred = model.predict(X_pred)[0]
        preds.append(y_pred)

    future_index = pd.date_range(df.index[-1] + timedelta(days=1), periods=horizon, freq="D")
    return pd.Series(preds, index=future_index), model

# ---------- Seasonality / Trend ----------
def seasonality_by_month(series):
    s = series["Sales"]
    return s.groupby(s.index.month).mean()

def rolling_trend(series, window=30):
    s = series["Sales"].copy()
    df = pd.DataFrame({"Sales": s})
    df["MA"] = s.rolling(window=window).mean()
    return df