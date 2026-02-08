import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings("ignore")




df = pd.read_csv("USGS_processed_2.csv")

X = df[['Month', 'Latitude', 'Longitude']]
y_mag = df['Magnitude']




X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y_mag, test_size=0.2, random_state=42)
mag_model = RandomForestRegressor(n_estimators=100, random_state=42)
mag_model.fit(X_train1, y_train1)
y_mag_pred = mag_model.predict(X_test1)



print("\nModel Evaluation:")


mae = mean_absolute_error(y_test1, y_mag_pred)
print(f"Earthquake Magnitude MAE: {mae:.3f}")


mse = mean_squared_error(y_test1, y_mag_pred)
print(f"Earthquake Magnitude MSE: {mse:.3f}")


rmse = np.sqrt(mse)
print(f"Earthquake Magnitude RMSE: {rmse:.3f}")


joblib.dump(mag_model, "quake_mag_model.pkl")

'''
Jan 2011 to Jun 2025
Model Evaluation:
Earthquake Magnitude MAE: 0.326
Earthquake Magnitude MSE: 0.205
Earthquake Magnitude RMSE: 0.453
'''

'''
Jan 2015 to Jun 2025
Model Evaluation:
Earthquake Magnitude MAE: 0.320
Earthquake Magnitude MSE: 0.198
Earthquake Magnitude RMSE: 0.445
'''


