# core/holiday_effects.py
import pandas as pd
import matplotlib.pyplot as plt

def holiday_summary(df, store_id=None):
    """
    Returns DataFrame with counts and avg sales for:
      - Promo days
      - StateHoliday (non-zero)
      - SchoolHoliday
      - Normal days
    If store_id provided, filters for that store.
    """
    d = df.copy()
    if store_id is not None:
        d = d[d["Store"] == store_id]

    # ensure string form
    d["StateHoliday"] = d["StateHoliday"].astype(str).fillna("0")

    stats = {}
    stats["Promo_count"] = int(d[d["Promo"] == 1].shape[0])
    stats["Promo_avg_sales"] = float(d[d["Promo"] == 1]["Sales"].mean()) if stats["Promo_count"] > 0 else 0.0

    stats["Holiday_count"] = int(d[d["StateHoliday"] != "0"].shape[0])
    stats["Holiday_avg_sales"] = float(d[d["StateHoliday"] != "0"]["Sales"].mean()) if stats["Holiday_count"] > 0 else 0.0

    stats["School_count"] = int(d[d["SchoolHoliday"] == 1].shape[0])
    stats["School_avg_sales"] = float(d[d["SchoolHoliday"] == 1]["Sales"].mean()) if stats["School_count"] > 0 else 0.0

    stats["Normal_count"] = int(d[(d["Promo"] == 0) & (d["StateHoliday"] == "0")].shape[0])
    stats["Normal_avg_sales"] = float(d[(d["Promo"] == 0) & (d["StateHoliday"] == "0")]["Sales"].mean()) if stats["Normal_count"] > 0 else 0.0

    return pd.Series(stats)

def plot_holiday_effects(ax, df, store_id=None):
    """
    ax: matplotlib axis
    df: full dataframe (train+store merged)
    draws two subplots on the given axis:
      - bar: average sales per category
      - bar: counts per category
    """
    s = holiday_summary(df, store_id=store_id)
    categories = ["Normal", "Promo", "Holiday", "School"]
    avg_sales = [s["Normal_avg_sales"], s["Promo_avg_sales"], s["Holiday_avg_sales"], s["School_avg_sales"]]
    counts = [s["Normal_count"], s["Promo_count"], s["Holiday_count"], s["School_count"]]

    ax.clear()
    ax.bar(categories, avg_sales, alpha=0.8)
    ax.set_title("Середні продажі: нормальні/промо/свята/школа")
    ax.set_ylabel("Середні продажі")
    # annotate
    for i, v in enumerate(avg_sales):
        ax.text(i, v + max(avg_sales) * 0.01, f"{v:.1f}", ha="center", fontsize=9)
