import numpy as np


def masked_mae(y_true, y_pred, mask=None):
    """Mean Absolute Error with optional mask for missing values."""
    if mask is not None:
        err = np.abs(y_true - y_pred)
        err = err[mask]
        return np.mean(err) if err.size > 0 else 0.0
    return np.mean(np.abs(y_true - y_pred))


def masked_rmse(y_true, y_pred, mask=None):
    """Root Mean Squared Error with optional mask."""
    if mask is not None:
        err = (y_true - y_pred) ** 2
        err = err[mask]
        return np.sqrt(np.mean(err)) if err.size > 0 else 0.0
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def masked_mape(y_true, y_pred, mask=None, eps=1e-8):
    """Mean Absolute Percentage Error with optional mask."""
    if mask is not None:
        y_true_m = y_true[mask]
        y_pred_m = y_pred[mask]
    else:
        y_true_m = y_true
        y_pred_m = y_pred
    denom = np.maximum(np.abs(y_true_m), eps)
    return np.mean(np.abs((y_true_m - y_pred_m) / denom)) * 100


def compute_all_metrics(y_true, y_pred, mask=None):
    """Compute MAE, RMSE, MAPE for flattened arrays.

    Args:
        y_true, y_pred: arrays of same shape
        mask: optional bool array, True = keep, False = masked (missing).
              METR-LA missing values encoded as 0 in raw data.
    """
    if mask is not None:
        yt = y_true[mask]
        yp = y_pred[mask]
    else:
        yt = y_true.reshape(-1)
        yp = y_pred.reshape(-1)
    return {
        "MAE": masked_mae(yt, yp),
        "RMSE": masked_rmse(yt, yp),
        "MAPE": masked_mape(yt, yp),
    }
