import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('Japan_processed.csv')

X = df[['Latitude', 'Longitude', 'Month']]
y = df[['Magnitude']]

# Split the data into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor for multi-output prediction
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Store evaluation metrics in a list
rf_metrics = [mae, mse, rmse, r2]
for idx, x in enumerate(rf_metrics):
    rf_metrics[idx] = round(rf_metrics[idx], 2)

# Create a list of headers and rows for tabulate
headers = ['Metric', 'Random Forest']
rows = [
    ['MAE'] + rf_metrics[0:1],
    ['MSE'] + rf_metrics[1:2],
    ['RMSE'] + rf_metrics[2:3],
    ['R² Score'] + rf_metrics[3:4]
]

# Generate and print the table
table = tabulate(rows, headers=headers, tablefmt='grid')
print(table)

# Save the model to a pickle file
with open('random_forest_regressor.pkl', 'wb') as pickle_file:
    pickle.dump(rf_model, pickle_file)