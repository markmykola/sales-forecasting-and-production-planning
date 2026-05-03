# core/utils.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_metrics(y_true, y_pred):
    """Обчислює MAE, RMSE, MAPE і Точність (%)"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        # trim or pad preds to match
        n = min(len(y_true), len(y_pred))
        y_true = y_true[:n]
        y_pred = y_pred[:n]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # exclude near-zero true values from MAPE (these give infinite/huge ratios)
    mask = y_true > 1e-8
    if mask.sum() > 0:
        mape = (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean()) * 100.0
    else:
        mape = 100.0

    accuracy = 100.0 - mape
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "mape": round(mape, 3), "accuracy": round(accuracy, 3)}
