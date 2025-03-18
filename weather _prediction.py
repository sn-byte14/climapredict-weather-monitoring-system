import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.ensemble import RandomForestClassifier  

# Load past 34 days of data  
df = pd.read_csv("weather_data.csv")  

# Convert timestamp to date (optional)  
df["Timestamp"] = pd.to_datetime(df["Timestamp"])  
df["Date"] = df["Timestamp"].dt.date  

# Drop timestamp, keep only numerical values  
df = df.groupby("Date").mean().reset_index()  

# Define features & target variables  
X = df[["Temperature (°C)", "Humidity (%)"]]  # Input Features  
y_temp = df["Temperature (°C)"].shift(-1).dropna()  # Target for temp prediction  
y_rain = df["Rain Detected"].shift(-1).dropna()  # Target for rain prediction  

# Train/Test Split  
X_train, X_test, y_temp_train, y_temp_test = train_test_split(X[:-1], y_temp, test_size=0.2, random_state=42)  
X_train_rain, X_test_rain, y_rain_train, y_rain_test = train_test_split(X[:-1], y_rain, test_size=0.2, random_state=42)  

# Train Linear Regression for Temp Prediction  
temp_model = LinearRegression()  
temp_model.fit(X_train, y_temp_train)  
temp_pred = temp_model.predict(X_test)  

# Train Random Forest for Rain Prediction  
rain_model = RandomForestClassifier(n_estimators=100, random_state=42)  
rain_model.fit(X_train_rain, y_rain_train)  
rain_pred = rain_model.predict(X_test_rain)  

# Show Predictions  
print(f"Predicted Temperature for Next Day: {temp_pred[-1]:.2f}°C")  
print(f"Predicted Rain (0 = No, 1 = Yes): {rain_pred[-1]}")  

# Plot Temperature Prediction  
plt.plot(y_temp_test.values, label="Actual Temp")  
plt.plot(temp_pred, label="Predicted Temp", linestyle="dashed")  
plt.legend()  
plt.title("Temperature Prediction")  
plt.show()
