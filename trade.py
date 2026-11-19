import os
import sqlite3
import pandas as pd

from sklearn.linear_model import LinearRegression

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Load data
conn = sqlite3.connect("regression_data.db")

train_df = pd.read_sql_query(
    "SELECT * FROM training_examples",
    conn,
)

test_df = pd.read_sql_query(
    "SELECT * FROM testing_examples",
    conn,
)

conn.close()

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

# Train model
X_train = train_df[features]
y_train = train_df["future_return"]

model = LinearRegression()
model.fit(X_train, y_train)

# Predict latest bar for each stock
latest = test_df.sort_values("ts_utc").groupby("symbol").tail(1).copy()

latest["prediction"] = model.predict(latest[features])

print(latest[["symbol", "prediction"]])

# Connect to Alpaca
client = TradingClient(os.environ["API_KEY"], os.environ["SECRET_KEY"], paper=True)

# Existing positions
positions = {p.symbol for p in client.get_all_positions()}

# Buy logic
for _, row in latest.iterrows():

    symbol = row["symbol"]
    prediction = row["prediction"]

    if prediction > 0.0005:

        if symbol in positions:
            print(f"Already own {symbol}")
            continue

        order = MarketOrderRequest(
            symbol=symbol,
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )

        client.submit_order(order_data=order)

        print(f"Bought {symbol} " f"Prediction={prediction:.6f}")
