# ui/main_ui.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- DESIGN LIBRARIES ---
import tkinter as tk
from tkinter import messagebox
try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *
    from ttkbootstrap.scrolled import ScrolledFrame
    DESIGN_MODE = True
except ImportError:
    import tkinter.ttk as ttk
    DESIGN_MODE = False
    print("Встановіть 'pip install ttkbootstrap' для кращого дизайну!")

# --- CORE IMPORTS ---
from core.forecasting import (
    load_rossmann, prepare_store_series,
    sarima_forecast, rf_forecast, auto_arima_forecast,
    seasonality_by_month, rolling_trend, sarima_conf_int
)
from core.planning import lp_production_plan
from core.utils import evaluate_metrics
from core.evaluation import plot_accuracy_bar, plot_error_bar
from core.holiday_effects import plot_holiday_effects, holiday_summary
from core.competition_effects import plot_competition_distance

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class ForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PlanMaster AI: Інтелектуальне планування")
        self.root.geometry("1280x800")
        
        # Налаштування стилю графіків під темну тему
        if DESIGN_MODE:
            plt.style.use('dark_background')
            self.colors = {
                'bg': '#2b3e50', 'fg': 'white', 
                'grid': '#444444', 'line1': '#4cc9f0', 'line2': '#f72585'
            }
        else:
            plt.style.use('ggplot')

        # Основний контейнер
        main_container = ttk.Frame(root, padding=10)
        main_container.pack(fill=BOTH, expand=True)

        # Вкладки
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill=BOTH, expand=True)

        self.tab_forecast = ttk.Frame(notebook, padding=10)
        self.tab_plan = ttk.Frame(notebook, padding=10)
        self.tab_accuracy = ttk.Frame(notebook, padding=10)

        notebook.add(self.tab_forecast, text=" 📈 Прогнозування ")
        notebook.add(self.tab_plan, text=" 🏭 Планування виробництва ")
        notebook.add(self.tab_accuracy, text=" 🎯 Точність моделей ")

        # Persistent Data
        self.df_full = None
        self.series = None 
        self.test_series = None 
        self.last_forecasts = {}
        self.last_models = {}
        self.last_plan = None
        self.last_plan_params = {}

        self._build_forecast_tab()
        self._build_plan_tab()
        self._build_accuracy_tab()

    # -------------- Helper: Unified Metrics Calculation --------------
    def _calculate_metrics_df(self, test_series, forecasts_dict):
        if not forecasts_dict or test_series is None:
            return pd.DataFrame()
            
        truth = test_series["Sales"]
        rows = []
        
        for name, fc in forecasts_dict.items():
            aligned = pd.concat([truth, fc], axis=1, join="inner").dropna()
            aligned.columns = ["Truth", "Pred"]
            
            if aligned.empty:
                continue

            y_true = aligned["Truth"].values
            y_pred = aligned["Pred"].values
            
            m = evaluate_metrics(y_true, y_pred)
            rows.append({
                "Модель": name,
                "MAE": m["mae"],
                "RMSE": m.get("rmse", 0),
                "MAPE (%)": m["mape"],
                "Точність (%)": m["accuracy"]
            })
            
        return pd.DataFrame(rows)

    # -------------- TAB 1: FORECAST --------------
    def _build_forecast_tab(self):
        # Панель керування
        control_frame = ttk.Labelframe(self.tab_forecast, text="Параметри прогнозу", padding=15, bootstyle="info")
        control_frame.pack(fill=X, pady=(0, 10))

        # Grid layout for inputs
        ttk.Label(control_frame, text="Store ID:", font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=5, sticky=W)
        self.store_entry = ttk.Entry(control_frame, width=10, bootstyle="secondary")
        self.store_entry.insert(0, "1")
        self.store_entry.grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="Горизонт (днів):", font=("Helvetica", 10, "bold")).grid(row=0, column=2, padx=5, sticky=W)
        self.horizon_entry = ttk.Entry(control_frame, width=10, bootstyle="secondary")
        self.horizon_entry.insert(0, "30")
        self.horizon_entry.grid(row=0, column=3, padx=5)

        ttk.Label(control_frame, text="Метод:", font=("Helvetica", 10, "bold")).grid(row=0, column=4, padx=5, sticky=W)
        self.method_combo = ttk.Combobox(control_frame, values=["SARIMA", "AutoARIMA", "Random Forest", "Порівняти всі", "Порівняти — найкраща"], state="readonly", width=25, bootstyle="primary")
        self.method_combo.set("Порівняти всі")
        self.method_combo.grid(row=0, column=5, padx=5)

        # Велика кнопка запуску
        btn_run = ttk.Button(control_frame, text="▶ Побудувати прогноз", command=self.run_forecast, bootstyle="primary")
        btn_run.grid(row=0, column=6, padx=20, sticky=E)

        # Графік
        plot_frame = ttk.Frame(self.tab_forecast, bootstyle="dark")
        plot_frame.pack(fill=BOTH, expand=True)
        
        self.fig_forecast = plt.Figure(figsize=(10, 5), dpi=100)
        self.fig_forecast.patch.set_facecolor('#2b3e50' if DESIGN_MODE else '#f0f0f0') # Match theme
        
        self.ax_forecast = self.fig_forecast.add_subplot(111)
        self.canvas_forecast = FigureCanvasTkAgg(self.fig_forecast, master=plot_frame)
        self.canvas_forecast.get_tk_widget().pack(fill=BOTH, expand=True, padx=2, pady=2)

    # -------------- TAB 2: PLANNING --------------
    def _build_plan_tab(self):
        # Верхня панель дій
        action_frame = ttk.Frame(self.tab_plan)
        action_frame.pack(fill=X, pady=(0, 10))

        # Ліва частина - Головна дія
        plan_btn = ttk.Button(action_frame, text="⚙ Розрахувати план", command=self.run_planning, bootstyle="success", width=20)
        plan_btn.pack(side=LEFT, padx=5)
        
        # Права частина - Аналітика
        ttk.Separator(action_frame, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10)
        
        ttk.Label(action_frame, text="Аналітика:", font=("Helvetica", 9)).pack(side=LEFT, padx=5)
        
        # Група кнопок аналітики (outline style для меншого візуального шуму)
        ttk.Button(action_frame, text="Сезонність", command=self.show_seasonality, bootstyle="outline-secondary").pack(side=LEFT, padx=2)
        ttk.Button(action_frame, text="Тренд", command=self.show_trend, bootstyle="outline-secondary").pack(side=LEFT, padx=2)
        ttk.Button(action_frame, text="Конкуренція", command=self.show_competition_effects, bootstyle="outline-secondary").pack(side=LEFT, padx=2)
        
        ttk.Separator(action_frame, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10)
        
        # Група Промо
        ttk.Button(action_frame, text="Промо (Графік)", command=self.show_promo_effects, bootstyle="outline-info").pack(side=LEFT, padx=2)
        ttk.Button(action_frame, text="Промо (Таймлайн)", command=self.show_promo_timeline, bootstyle="outline-info").pack(side=LEFT, padx=2)
        
        # Другий ряд кнопок (Звіти)
        report_frame = ttk.Frame(self.tab_plan)
        report_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Button(report_frame, text="📊 Довірчий інтервал", command=self.show_confidence, bootstyle="warning").pack(side=LEFT, padx=5)
        ttk.Button(report_frame, text="💰 Фінансовий звіт", command=self.show_financial_report, bootstyle="danger").pack(side=LEFT, padx=5)

        # Графік
        self.fig_plan = plt.Figure(figsize=(10, 5), dpi=100)
        self.fig_plan.patch.set_facecolor('#2b3e50' if DESIGN_MODE else '#f0f0f0')
        
        self.ax_plan = self.fig_plan.add_subplot(111)
        self.canvas_plan = FigureCanvasTkAgg(self.fig_plan, master=self.tab_plan)
        self.canvas_plan.get_tk_widget().pack(fill=BOTH, expand=True)

    # -------------- TAB 3: ACCURACY --------------
    def _build_accuracy_tab(self):
        # Верхня панель
        top_frame = ttk.Frame(self.tab_accuracy)
        top_frame.pack(fill=X, pady=10)
        
        ttk.Button(top_frame, text="🔄 Оновити метрики", command=self.plot_accuracy_chart, bootstyle="primary-outline").pack(side=LEFT, padx=10)

        # Графіки (ліворуч) і Таблиця (праворуч) або Верх/Низ
        # Зробимо спліт: зверху графіки, знизу таблиця
        
        graph_frame = ttk.Frame(self.tab_accuracy)
        graph_frame.pack(fill=BOTH, expand=True, pady=5)
        
        self.fig_acc, (self.ax_acc, self.ax_err) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig_acc.patch.set_facecolor('#2b3e50' if DESIGN_MODE else '#f0f0f0')
        
        self.canvas_acc = FigureCanvasTkAgg(self.fig_acc, master=graph_frame)
        self.canvas_acc.get_tk_widget().pack(fill=BOTH, expand=True)

        # Таблиця
        table_frame = ttk.Labelframe(self.tab_accuracy, text="Детальна статистика", padding=10, bootstyle="secondary")
        table_frame.pack(fill=X, pady=10, padx=5)

        columns = ("Модель", "MAE", "RMSE", "MAPE (%)", "Точність (%)")
        self.metrics_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=5, bootstyle="info")
        
        for col in columns:
            self.metrics_table.heading(col, text=col)
            self.metrics_table.column(col, width=120, anchor="center")
        
        self.metrics_table.pack(fill=X)

    # -------------- ACTIONS --------------

    def _update_plot_style(self, ax, title, xlabel=None, ylabel=None):
        """Допоміжна функція для красивого оформлення графіків"""
        if DESIGN_MODE:
            ax.set_facecolor('#2b3e50')
            ax.grid(color='#444444', linestyle='--', linewidth=0.5)
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            # remove spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
        
        ax.set_title(title, fontsize=12, pad=10)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)

    def run_forecast(self):
        try:
            store_id = int(self.store_entry.get())
            horizon = int(self.horizon_entry.get())
            method = self.method_combo.get()

            df = load_rossmann(os.path.join(BASE_DIR, "data/train.csv"),
                               os.path.join(BASE_DIR, "data/store.csv"))
            self.df_full = df
            series = prepare_store_series(df, store_id)
            self.series = series 
            
            train_series = series.iloc[:-horizon]
            test_series = series.iloc[-horizon:]
            self.test_series = test_series 

            temp_forecasts = {}
            temp_models = {}
            candidates = []
            
            if method in ("SARIMA", "Порівняти всі", "Порівняти — найкраща"): candidates.append("SARIMA")
            if method in ("AutoARIMA", "Порівняти всі", "Порівняти — найкраща"): candidates.append("AutoARIMA")
            if method in ("Random Forest", "Порівняти всі", "Порівняти — найкраща"): candidates.append("Random Forest")
            
            # Progress bar simulation (optional visual)
            self.root.config(cursor="watch")
            self.root.update()

            if "SARIMA" in candidates:
                fc, mod = sarima_forecast(train_series, horizon=horizon, future_exog_df=test_series)
                temp_forecasts["SARIMA"] = fc.clip(lower=0); temp_models["SARIMA"] = mod

            if "AutoARIMA" in candidates:
                fc, mod = auto_arima_forecast(train_series, horizon=horizon, future_exog_df=test_series)
                temp_forecasts["AutoARIMA"] = fc.clip(lower=0); temp_models["AutoARIMA"] = mod

            if "Random Forest" in candidates:
                fc, mod = rf_forecast(train_series, horizon=horizon, future_exog_df=test_series)
                temp_forecasts["Random Forest"] = fc.clip(lower=0); temp_models["Random Forest"] = mod

            self.root.config(cursor="")

            final_forecasts = temp_forecasts
            final_models = temp_models

            if method == "Порівняти — найкраща":
                metrics_df = self._calculate_metrics_df(test_series, temp_forecasts)
                if not metrics_df.empty:
                    best_row = metrics_df.loc[metrics_df["MAPE (%)"].idxmin()]
                    best_name = best_row["Модель"]
                    messagebox.showinfo("Auto-Select", f"Найкраща модель: {best_name} (MAPE: {best_row['MAPE (%)']:.2f}%)")
                    final_forecasts = {best_name: temp_forecasts[best_name]}
                    final_models = {best_name: temp_models.get(best_name)}

            self.last_forecasts = final_forecasts
            self.last_models = final_models

            # Plotting
            self.ax_forecast.clear()
            self._update_plot_style(self.ax_forecast, f"Прогноз продажів: Магазин {store_id}", "Дата", "Продажі")
            
            hist = series["Sales"].iloc[-180:] 
            self.ax_forecast.plot(hist.index, hist.values, label="Історія", color="#a0a0a0", alpha=0.7)
            self.ax_forecast.plot(self.test_series.index, self.test_series["Sales"], label="Факт (Test)", color="#ff5252", marker=".", linewidth=0)
            
            colors = ['#4cc9f0', '#f72585', '#ffd60a'] # Custom neon colors
            for i, (name, fc) in enumerate(final_forecasts.items()):
                c = colors[i % len(colors)]
                self.ax_forecast.plot(fc.index, fc.values, linestyle="-", linewidth=2, label=name, color=c)
                
            self.ax_forecast.legend(facecolor='#2b3e50', edgecolor='white', labelcolor='white')
            self.canvas_forecast.draw()

        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Помилка", str(e))

    def run_planning(self):
        if not self.last_forecasts:
            messagebox.showinfo("Увага", "Спочатку побудуйте прогноз!")
            return
        
        name, forecast = next(iter(self.last_forecasts.items()))
        demand = forecast.clip(lower=0).round().astype(int)
        
        params = {"prod_cost": 2.0, "hold_cost": 0.05, "capacity": 15000.0, "initial_inventory": 5000.0, "backorder_cost": 10.0}
        self.last_plan_params = params
        
        plan = lp_production_plan(demand, params)
        self.last_plan = plan

        self.ax_plan.clear()
        self._update_plot_style(self.ax_plan, f"Виробничий план (База: {name})")
        
        self.ax_plan.plot(plan.index, plan["demand"], label="Попит", linestyle="--", marker='o', markersize=4, color="#a0a0a0", zorder=3)
        self.ax_plan.bar(plan.index, plan["produce"], label="Виробництво", color="#4cc9f0", alpha=0.6, width=0.8, zorder=2)
        
        if "backorder" in plan.columns and plan["backorder"].sum() > 0:
             self.ax_plan.plot(plan.index, plan["backorder"], label="ДЕФІЦИТ!", color="#ff0000", linewidth=2, zorder=5)

        self.ax_plan.legend(facecolor='#2b3e50', edgecolor='white', labelcolor='white')
        self.canvas_plan.draw()

    def show_financial_report(self):
        if self.last_plan is None:
            messagebox.showinfo("Увага", "Розрахуйте план для перегляду звіту.")
            return

        df = self.last_plan
        p = self.last_plan_params
        total_prod = df["produce"].sum(); cost_prod = total_prod * p["prod_cost"]
        total_hold = df["inventory_end"].sum(); cost_hold = total_hold * p["hold_cost"]
        total_back = df["backorder"].sum() if "backorder" in df.columns else 0; cost_back = total_back * p["backorder_cost"]
        total_cost = cost_prod + cost_hold + cost_back
        
        report = (
            f"💰 ФІНАНСОВИЙ АНАЛІЗ\n"
            f"=====================\n"
            f"ВАРТІСТЬ ПЛАНУ: {total_cost:,.2f} у.о.\n\n"
            f"🏭 Виробництво: {cost_prod:,.2f} ({int(total_prod)} од.)\n"
            f"📦 Зберігання:  {cost_hold:,.2f} ({int(total_hold)} од*дн)\n"
            f"⚠  Штрафи:      {cost_back:,.2f} ({int(total_back)} од.)"
        )
        messagebox.showinfo("Фінанси", report)

    # --- PLOTTING HELPERS ---
    def show_seasonality(self):
        if self.series is None: return
        monthly = seasonality_by_month(self.series)
        self.ax_plan.clear()
        self._update_plot_style(self.ax_plan, "Сезонність (по місяцях)", "Місяць", "Продажі")
        self.ax_plan.bar(monthly.index, monthly.values, color="#f72585", alpha=0.8)
        self.canvas_plan.draw()

    def show_trend(self):
        if self.series is None: return
        df = rolling_trend(self.series, window=30)
        self.ax_plan.clear()
        self._update_plot_style(self.ax_plan, "Тренд продажів (MA30)")
        col_name = "Sales" if "Sales" in df.columns else df.columns[0]
        self.ax_plan.plot(df.index, df[col_name], label="Факт", alpha=0.4, color="white")
        self.ax_plan.plot(df.index, df["MA"], label="MA30 Trend", linewidth=3, color="#ffd60a")
        self.ax_plan.legend(facecolor='#2b3e50', labelcolor='white')
        self.canvas_plan.draw()

    def show_promo_effects(self):
        if self.df_full is None: return
        try: store_id = int(self.store_entry.get())
        except: store_id = None
        self.ax_plan.clear()
        self._update_plot_style(self.ax_plan, "Ефект Промо та Свят")
        plot_holiday_effects(self.ax_plan, self.df_full, store_id=store_id)
        self.canvas_plan.draw()

    def show_promo_timeline(self):
        if self.series is None: return
        self.ax_plan.clear()
        self._update_plot_style(self.ax_plan, "Таймлайн подій (останній рік)")
        df = self.series.iloc[-365:].copy()
        self.ax_plan.plot(df.index, df["Sales"], linewidth=1, label="Продажі", alpha=0.6, color="white")
        
        self.ax_plan.scatter(df[df["Promo"]==1].index, df[df["Promo"]==1]["Sales"], marker="o", label="Промо", color="#ffd60a", zorder=3)
        self.ax_plan.scatter(df[df["StateHoliday_Numeric"]!=0].index, df[df["StateHoliday_Numeric"]!=0]["Sales"], marker="D", label="Свято", color="#f72585", zorder=4)
        
        self.ax_plan.legend(facecolor='#2b3e50', labelcolor='white')
        self.canvas_plan.draw()

    def show_competition_effects(self):
        if self.df_full is None: return
        self.ax_plan.clear()
        self._update_plot_style(self.ax_plan, "Вплив конкуренції")
        try: plot_competition_distance(self.ax_plan, self.df_full)
        except Exception as e: messagebox.showerror("Error", str(e))
        self.canvas_plan.draw()

    def show_confidence(self):
        if not self.last_forecasts: return
        metrics_df = self._calculate_metrics_df(self.test_series, self.last_forecasts)
        best = metrics_df.loc[metrics_df["MAPE (%)"].idxmin()]["Модель"] if not metrics_df.empty else next(iter(self.last_forecasts.keys()))

        fc = self.last_forecasts[best]; fitted = self.last_models.get(best)
        conf = None
        try: conf = sarima_conf_int(fitted, steps=len(fc), future_exog_df=self.test_series)
        except: pass

        self.ax_plan.clear()
        self._update_plot_style(self.ax_plan, f"{best}: Довірчий інтервал")
        hist = self.series["Sales"].iloc[-90:]
        self.ax_plan.plot(hist.index, hist.values, label="Історія", color="white", alpha=0.5)
        self.ax_plan.plot(fc.index, fc.values, label="Прогноз", color="#4cc9f0")
        if conf is not None:
            self.ax_plan.fill_between(fc.index, conf.iloc[:, 0].clip(lower=0), conf.iloc[:, 1], color="#4cc9f0", alpha=0.2)
        self.canvas_plan.draw()

    # --- ACCURACY ---
    def compute_metrics_df(self):
        return self._calculate_metrics_df(self.test_series, self.last_forecasts)

    def plot_accuracy_chart(self):
        df = self.compute_metrics_df()
        if df is None or df.empty: return

        for row in self.metrics_table.get_children(): self.metrics_table.delete(row)
        for _, r in df.iterrows():
            vals = [r["Модель"], r["MAE"]]
            if "RMSE" in df.columns: vals.append(r["RMSE"])
            vals.extend([r["MAPE (%)"], r["Точність (%)"]])
            
            # Динамічне налаштування колонок, якщо RMSE з'явилось/зникло
            cols = ("Модель", "MAE", "RMSE", "MAPE (%)", "Точність (%)") if "RMSE" in df.columns else ("Модель", "MAE", "MAPE (%)", "Точність (%)")
            self.metrics_table.configure(columns=cols)
            for c in cols: self.metrics_table.heading(c, text=c)

            self.metrics_table.insert("", "end", values=vals)

        self.ax_acc.clear(); self._update_plot_style(self.ax_acc, "Точність (100 - MAPE)")
        plot_accuracy_bar(self.ax_acc, df)
        
        self.ax_err.clear(); self._update_plot_style(self.ax_err, "Помилка (MAPE)")
        plot_error_bar(self.ax_err, df)
        
        self.canvas_acc.draw()

if __name__ == "__main__":
    # ВИКОРИСТОВУЄМО ТЕМУ SUPERHERO (Темна) АБО FLATLY (Світла)
    if DESIGN_MODE:
        app = ttk.Window(themename="superhero") # Спробуйте "darkly", "superhero" або "flatly"
    else:
        app = tk.Tk()
    
    ForecastApp(app)
    app.mainloop()
