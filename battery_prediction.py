import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from the CSV file
data = pd.read_csv("data/battery_data.csv")

# Select input (X) and output (y)
X = data[["screen_time"]]
y = data["battery_used"]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict battery usage for the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model - Battery Usage Prediction")
print("---------------------------------------------------")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.3f}")

# Display actual and predicted values
results = pd.DataFrame({
    "screen_time": X_test["screen_time"].values,
    "actual_battery_used": y_test.values,
    "predicted_battery_used": y_pred.round(2)
}).sort_values("screen_time")

print("\nActual vs Predicted Battery Usage:")
print(results.to_string(index=False))

# Predict battery usage for every screen-time value in the CSV
data["predicted_battery_used"] = model.predict(X).round(2)
data.to_csv("data/battery_predictions.csv", index=False)
print("\nPredictions saved to data/battery_predictions.csv")

# Visualize the data and regression line
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.xlabel("Screen Time (minutes)")
plt.ylabel("Battery Used (%)")
plt.title("Screen Time vs Battery Usage")
plt.legend()
plt.show()
