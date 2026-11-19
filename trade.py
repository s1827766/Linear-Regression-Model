# Imports

import os
import sqlite3
import pandas as pd

from sklearn.linear_model import LinearRegression

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Load API keys

load_dotenv()

API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")


# Connect to Alpaca

client = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=True,  # change to False for live trading
)


# Load regression dataset

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


# Define features

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


# Train Linear Regression model

X_train = train_df[features]
y_train = train_df["future_return"]

model = LinearRegression()
model.fit(X_train, y_train)


# Get latest row for each stock

latest = test_df.sort_values("ts_utc").groupby("symbol").tail(1).copy()


# Generate predictions

latest["prediction"] = model.predict(latest[features])

print("\nPredictions")
print(latest[["symbol", "prediction"]])


# Get current positions

positions = {p.symbol: p for p in client.get_all_positions()}


# Trading thresholds

BUY_THRESHOLD = 0.0005
SELL_THRESHOLD = -0.0005


# Process signals

for _, row in latest.iterrows():

    symbol = row["symbol"]
    prediction = row["prediction"]

    print(f"\n{symbol} | Prediction = {prediction:.6f}")

    # SELL

    if prediction < SELL_THRESHOLD:

        if symbol in positions:

            qty = abs(int(float(positions[symbol].qty)))

            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )

            client.submit_order(order_data=order)

            print(f"SELL {qty} shares of {symbol}")

        else:

            print(f"HOLD {symbol} (nothing to sell)")

    # BUY

    elif prediction > BUY_THRESHOLD:

        if symbol not in positions:

            order = MarketOrderRequest(
                symbol=symbol,
                qty=1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )

            client.submit_order(order_data=order)

            print(f"BUY 1 share of {symbol}")

        else:

            print(f"HOLD {symbol} (already owned)")

    # HOLD

    else:

        print(f"HOLD {symbol}")


# Summary

print("\nTrading run complete.")
