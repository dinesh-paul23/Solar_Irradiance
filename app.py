import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import datetime

# Suppress warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# Load dataset (same directory as solar.py)
df = pd.read_csv("mdu_data.csv")
df = df.fillna(0)

# Feature and target columns
cols = [0, 1, 2, 3, 4]  # Adjust if column names are available for clarity
X = df[df.columns[cols]].values
Y_temp = df[df.columns[5]].values.ravel()
Y_ghi = df[df.columns[6]].values.ravel()

# Split dataset
x_train, x_test, y_temp_train, y_temp_test = train_test_split(X, Y_temp, random_state=42)
_, _, y_ghi_train, y_ghi_test = train_test_split(X, Y_ghi, random_state=42)

# Train models
temp_model = RandomForestRegressor()
ghi_model = RandomForestRegressor()

temp_model.fit(x_train, y_temp_train)
ghi_model.fit(x_train, y_ghi_train)

# Predict for current time +15 minutes
current_time = datetime.datetime.now() + datetime.timedelta(minutes=15)
time_updated = current_time.strftime("%Y-%m-%d %H:%M")

now = current_time.strftime("%Y,%m,%d,%H,%M")
now = [int(i) for i in now.split(",")]

temp = temp_model.predict([now])
ghi = ghi_model.predict([now])

# Power calculation
f = 0.18 * 7.4322 * ghi
insi = 0.05 * (temp - 25)
midd = 1 - insi
power = f * midd

# Accuracy (optional)
temp_accuracy = max(0, min(100, r2_score(y_temp_test, temp_model.predict(x_test)) * 100))
ghi_accuracy = max(0, min(100, r2_score(y_ghi_test, ghi_model.predict(x_test)) * 100))

# Output results
print(f"Prediction at {time_updated}:")
print(f"  Temperature: {temp[0]:.2f}°C (Accuracy: {temp_accuracy:.2f}%)")
print(f"  GHI: {ghi[0]:.2f} W/m² (Accuracy: {ghi_accuracy:.2f}%)")
print(f"  Power: {power[0]:.2f} W")
