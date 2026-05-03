# run_app.py
from ui.main_ui import ForecastApp
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ForecastApp(root)
    root.mainloop()
