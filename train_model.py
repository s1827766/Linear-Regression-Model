import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

conn = sqlite3.connect("regression_data.db")

train_df = pd.read_sql_query("SELECT * FROM training_examples", conn)
test_df = pd.read_sql_query("SELECT * FROM testing_examples", conn)

conn.close()
# X_train = train_df[
#     ["ma5", "ma10", "ma20", "ma50", "return_1", "return_3", "return_6", "volatility"]
# ]

features = [
    "ma5_ratio",
    "ma10_ratio",
    "ma20_ratio",
    "ma50_ratio",
    "return_1",
    "return_3",
    "return_6",
    "volatility",
    "volume_change",
]

corrs = train_df[features + ["future_return"]].corr()["future_return"]

print("\nCorrelations with future_return")
print(corrs.sort_values(key=abs, ascending=False))

X_train = train_df[features]
X_test = test_df[features]
y_train = train_df["future_return"]

# X_test = test_df[["return_1", "return_3", "return_6"]]
y_test = test_df["future_return"]

# X_test = test_df[
#     ["ma5", "ma10", "ma20", "ma50", "return_1", "return_3", "return_6", "volatility"]
# ]

model = LinearRegression()
model.fit(X_train, y_train)

print("\nTraining Performance")
print("----------------")
print("Train R²:", round(model.score(X_train, y_train), 4))
print("Test  R²:", round(model.score(X_test, y_test), 4))

coef_df = pd.DataFrame({"Feature": X_train.columns, "Coefficient": model.coef_})

print("\nFeature Coefficients")
print(coef_df.sort_values("Coefficient", key=lambda x: abs(x), ascending=False))

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Results")
print("----------------")
print("Mean Absolute Error:", round(mae, 4))
print("R-squared:", round(r2, 4))

results = test_df.copy()
results["prediction"] = predictions

print("\nSample Predictions")
print(results[["symbol", "close", "future_return", "prediction"]].head(20))

results["buy_signal"] = results["prediction"] > 0.0004

print("\nBuy Signals")
print(results[["symbol", "close", "prediction", "buy_signal"]].head(20))


plt.figure(figsize=(8, 6))

plt.scatter(y_test, predictions, alpha=0.4)

plt.xlabel("Actual Return")
plt.ylabel("Predicted Return")
plt.title("Actual vs Predicted Returns")

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")

plt.show()


results = test_df.copy()
results["prediction"] = predictions

plt.figure(figsize=(12, 6))

plt.plot(results["future_return"].values[:200], label="Actual")

plt.plot(results["prediction"].values[:200], label="Predicted")

plt.legend()
plt.title("Actual vs Predicted Returns")
plt.xlabel("Observation")
plt.ylabel("Return")

plt.show()


features = X_train.columns

plt.figure(figsize=(8, 5))

plt.bar(features, model.coef_)

plt.title("Feature Coefficients")
plt.ylabel("Coefficient")

plt.show()
