import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import warnings

warnings.filterwarnings("ignore")

# ======================================
# Step 1: Load & Set Features and Target
# ======================================

# Load your CSV
df = pd.read_csv("USGS_processed.csv")
#print(df.head())

# Features and targets
X = df[['Month', 'Latitude', 'Longitude']]
y_mag = df['Magnitude']

# ===========================
# Step 2: Train Models
# ===========================

# Magnitude prediction model
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y_mag, test_size=0.2, random_state=42)
mag_model = RandomForestRegressor(n_estimators=100, random_state=42)
mag_model.fit(X_train1, y_train1)
y_mag_pred = mag_model.predict(X_test1)

# ===========================
# Step 3: Evaluate Models
# ===========================

print("\nModel Evaluation:")
print(f"Earthquake Magnitude RMSE: {mean_squared_error(y_test1, y_mag_pred):.3f}")

# ===========================
# Step 4: Save Models (Optional)
# ===========================

joblib.dump(mag_model, "quake_mag_model.pkl")

'''
Model Evaluation:
Earthquake Magnitude RMSE: 0.189
'''