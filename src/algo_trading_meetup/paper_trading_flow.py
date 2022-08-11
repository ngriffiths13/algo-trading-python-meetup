"""A Prefect flow to execute paper trading.

This is just a quick and dirty example of how you might deploy a paper trading flow
using Prefect and the Alpaca trading API. It runs the model trained in the Jupyter
Notebook to get a trading signal for today.
"""
import asyncio
from datetime import datetime, timedelta

import joblib
import pandas as pd
from alpaca.data.enums import Adjustment
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from finta import TA
from prefect import flow, get_run_logger, task
from prefect.blocks.system import Secret, String


@task
async def pull_pricing_data():
    """Pull the last 50 days of price data to create our dataset."""
    api_key = await String.load("alpaca-api-key")
    secret_key = await Secret.load("alpaca-secret-key")

    client = StockHistoricalDataClient(api_key.value, secret_key.get())
    response = client.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=["AAPL"],
            start=datetime.today() - timedelta(days=50),
            end=datetime.today(),
            timeframe=TimeFrame.Day,
            adjustment=Adjustment.ALL,
        )
    )
    df = response.df
    df.index = [i[1] for i in df.index]
    return df


@task
async def create_features(x: pd.DataFrame):
    """Create all of our model features."""
    x["ssma"] = TA.SSMA(x, period=30)
    x["msd"] = TA.MSD(x, period=30)
    x["adx"] = TA.ADX(x, period=30)
    x[["bband_upper", "bband_middle", "bband_lower"]] = TA.BBANDS(x, period=30)
    x["rsi"] = TA.RSI(x, period=30)
    x["cross_lower_band"] = x["ssma"] < x["bband_lower"]
    x["cross_back_lower_band"] = (
        ~x["cross_lower_band"].shift(1).fillna(False) & x["cross_lower_band"]
    )
    features = [
        "ssma",
        "msd",
        "adx",
        "rsi",
        "bband_upper",
        "bband_middle",
        "bband_lower",
        "cross_lower_band",
        "cross_back_lower_band",
    ]

    return x[-1:][features]


@task
async def make_preds(x: pd.DataFrame):
    """Load in our model and make predictions."""
    model = joblib.load("./models/rf_aapl_model.pkl")
    pred = model.predict(x)
    logger = get_run_logger()
    prediction = pred[0]
    if prediction == 1:
        logger.info("Generated Buy Signal")
    elif prediction == -1:
        logger.info("Generated Sell Signal")
    else:
        logger.info("Generated Hold Signal")
    return prediction


@task
async def make_trades(signal: int):
    """Take the signal generated from our model and make a trading decision."""
    logger = get_run_logger()
    api_key = await String.load("alpaca-api-key")
    secret_key = await Secret.load("alpaca-secret-key")
    position = 0
    trading_client = TradingClient(
        api_key.value, secret_key.get(), paper=True
    )  # , url_override="http://paper-api.alpaca.markets")
    current_positions = trading_client.get_all_positions()
    if len(current_positions) > 0:
        position = 1
        logger.info("Currently holding 100 shares of AAPL")

    if position == 0 and signal == 1:
        # preparing orders
        market_order_data = MarketOrderRequest(
            symbol="AAPL", qty=100, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
        )
        logger.info("Sending Market Order to buy 100 shares of AAPL")

        # Market order
        trading_client.submit_order(order_data=market_order_data)

    if position == 1 and signal == -1:
        trading_client.close_all_positions(cancel_orders=True)
        logger.info("Sending Market Order to sell 100 shares of AAPL")


@flow(name="AAPL Trading Flow")
async def execute_aapl_trades():
    """A trading flow to execute AAPL trades."""
    data = await pull_pricing_data()
    features = await create_features(data)
    signal = await make_preds(features)
    await make_trades(signal)


if __name__ == "__main__":
    asyncio.run(execute_aapl_trades())
