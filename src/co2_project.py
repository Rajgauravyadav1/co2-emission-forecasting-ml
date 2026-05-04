import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Greenhouse Gas Emissions Dataset
df = pd.read_csv('co2_emissions.csv')

# Preprocessing: Focus on per-capita CO2 features
# Identify high-emission regions by historical average
high_emitters = df.groupby('country_region')['per_capita_co2'].mean().nlargest(5).index.tolist()

# Prepare features (Year and Country)
df['country_cat'] = df['country_region'].astype('category').cat.codes
X = df[['year', 'country_cat']]
y = df['per_capita_co2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Random Forest for Forecasting
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Calculate Judging Metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
r2 = r2_score(y_test, predictions)

print(f"Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, R2={r2:.2f}")

# Save future predictions for use in app.py
future_years = pd.DataFrame({'year': range(2025, 2036), 'country_cat': X_test.iloc[0]['country_cat']})
future_preds = model.predict(future_years)
future_years['predicted_co2'] = future_preds
future_years.to_csv('future_predictions.csv', index=False)
