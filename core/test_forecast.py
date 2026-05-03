from core.forecasting import load_rossmann, prepare_store_series, sarima_forecast, rf_forecast

df = load_rossmann("data/train.csv", "data/store.csv")
series = prepare_store_series(df, store_id=1)

sarima_fc, _ = sarima_forecast(series, horizon=30)
rf_fc, _ = rf_forecast(series, horizon=30)

print("SARIMA forecast head:")
print(sarima_fc.head())

print("\nRandomForest forecast head:")
print(rf_fc.head())
