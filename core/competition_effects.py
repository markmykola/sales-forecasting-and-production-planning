# core/competition_effects.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_competition_distance(df):
    """
    Аналізує вплив відстані до конкурента на продажі.
    Групує відстані (біни) та розраховує середні продажі для кожної групи.
    
    Повертає pd.Series з середніми продажами, індексований за назвами груп.
    """
    if 'CompetitionDistance' not in df.columns:
        return pd.Series(dtype=float)

    # Спочатку візьмемо унікальні магазини, оскільки відстань однакова для магазину
    store_df = df.drop_duplicates(subset=['Store']).copy()
    
    # Обробка пропущених значень (магазини без даних про конкурентів)
    # Ми можемо припустити, що це "дуже далеко" (np.inf)
    store_df['CompetitionDistance'] = store_df['CompetitionDistance'].fillna(np.inf)

    # Визначаємо групи (біни) у метрах
    bins = [0, 1000, 5000, 20000, np.inf]
    labels = ["< 1км", "1-5км", "5-20км", "> 20км (або немає)"]
    
    store_df['DistanceBin'] = pd.cut(store_df['CompetitionDistance'], bins=bins, labels=labels, right=False)
    
    # Тепер об'єднуємо цю інформацію про групи назад до повного df,
    # щоб отримати всі транзакції продажів
    df_merged = df.merge(store_df[['Store', 'DistanceBin']], on='Store', how='left')
    
    # Розраховуємо середні продажі для кожної групи
    avg_sales = df_merged.groupby('DistanceBin')['Sales'].mean()
    
    return avg_sales

def plot_competition_distance(ax, df):
    """
    Малює стовпчикову діаграму середніх продажів за відстанню 
    до конкурента на наданій осі (ax).
    """
    ax.clear()
    avg_sales = analyze_competition_distance(df)
    
    if avg_sales.empty:
        ax.text(0.5, 0.5, "Дані 'CompetitionDistance' відсутні", 
                ha="center", va="center", transform=ax.transAxes)
        return

    ax.bar(avg_sales.index, avg_sales.values, color="teal", alpha=0.8)
    ax.set_title("Середні продажі vs. Відстань до конкурента")
    ax.set_ylabel("Середні продажі")
    ax.set_xlabel("Відстань до найближчого конкурента")
    
    # Додаємо мітки
    for i, v in enumerate(avg_sales.values):
        ax.text(i, v + (avg_sales.max() * 0.01), f"{v:.0f}", ha="center", fontsize=9)
        
    ax.grid(axis="y", linestyle="--", alpha=0.3)