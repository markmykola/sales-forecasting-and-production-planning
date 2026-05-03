# core/evaluation.py
import pandas as pd
import matplotlib.pyplot as plt

def compute_metrics_df_from_dict(metrics_dict):
    """
    Accepts dict like {"SARIMA": {"MAE":..., "RMSE":..., "MAPE (%)":...}, ...}
    Returns DataFrame with columns ["Модель","MAE","RMSE","MAPE (%)","Точність (%)"]
    """
    df = pd.DataFrame(metrics_dict).T.reset_index()
    df.rename(columns={"index": "Модель"}, inplace=True)
    if "MAPE (%)" in df.columns:
        df["Точність (%)"] = 100.0 - df["MAPE (%)"]
    else:
        df["MAPE (%)"] = 0.0
        df["Точність (%)"] = 0.0
    return df.round(3)

def plot_accuracy_bar(ax, metrics_df):
    """
    Build accuracy bar (Точність (%)). Accepts DataFrame with either:
      - column "Точність (%)" or
      - column "MAPE (%)" (will compute 100 - MAPE)
      - column "Accuracy (%)" (will use it)
    and column "Модель" (or "Model").
    """
    ax.clear()
    df = metrics_df.copy()

    # rename if accidental english columns used
    if "Model" in df.columns and "Модель" not in df.columns:
        df.rename(columns={"Model": "Модель"}, inplace=True)

    if "Точність (%)" not in df.columns:
        if "Accuracy (%)" in df.columns:
            df["Точність (%)"] = df["Accuracy (%)"]
        elif "MAPE (%)" in df.columns:
            df["Точність (%)"] = 100.0 - df["MAPE (%)"]
        else:
            df["Точність (%)"] = 0.0

    # ensure non-negative
    df["Точність (%)"] = df["Точність (%)"].clip(lower=0.0)

    ax.bar(df["Модель"], df["Точність (%)"], color="skyblue")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Точність (%)")
    ax.set_title("Точність моделей (100 - MAPE)")
    for i, v in enumerate(df["Точність (%)"]):
        if pd.isna(v):
            continue
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

def plot_error_bar(ax, metrics_df):
    """
    Plot error (MAPE %). Accepts DataFrame with "MAPE (%)" and "Модель".
    """
    ax.clear()
    df = metrics_df.copy()
    if "Model" in df.columns and "Модель" not in df.columns:
        df.rename(columns={"Model": "Модель"}, inplace=True)
    if "MAPE (%)" not in df.columns:
        df["MAPE (%)"] = 0.0

    ylim = max(df["MAPE (%)"].max() * 1.2, 10) if len(df) > 0 else 10
    ax.bar(df["Модель"], df["MAPE (%)"], color="salmon")
    ax.set_ylim(0, ylim)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Середня відносна похибка (MAPE)")
    for i, v in enumerate(df["MAPE (%)"]):
        if pd.isna(v):
            continue
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

