"""A Prefect flow to execute paper trading.

This is just a quick and dirty example of how you might deploy a paper trading flow
using Prefect and the Alpaca trading API. It runs the model trained in the Jupyter
Notebook to get a trading signal for today.
"""
import asyncio
from prefect import task, flow
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import Adjustment
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame
from prefect.blocks.system import String
import pandas as pd
from prefect.blocks.system import Secret
from finta import TA
import joblib

@task
async def pull_pricing_data():
    api_key = await String.load("alpaca-api-key")
    secret_key = await Secret.load("alpaca-secret-key")

    client = StockHistoricalDataClient(api_key.value, secret_key.get())
    response = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=["AAPL"],
                                                 start=datetime.today() - timedelta(days=50),
                                                 end=datetime.today(),
                                                 timeframe=TimeFrame.Day, 
                                                 adjustment=Adjustment.ALL))
    df = response.df
    df.index = [i[1] for i in df.index]
    if max(df.index) != datetime.today():
        exit(0)
    return df


@task
async def create_features(x: pd.DataFrame):
    x["ssma"] = TA.SSMA(x, period=30)
    x["msd"] = TA.MSD(x, period=30)
    x["adx"] = TA.ADX(x, period=30)
    x[["bband_upper", "bband_middle", "bband_lower"]] = TA.BBANDS(x, period=30)
    x["cross_lower_band"] = x["ssma"] < x["bband_lower"]
    x["cross_back_lower_band"] = (~x["cross_lower_band"].shift(1).fillna(False) & x["cross_lower_band"])
    features = ["ssma", "msd", "rsi", "adx", "bband_upper", "bband_middle", "bband_lower", "cross_lower_band", "cross_back_lower_band"]

    return x[-1:][features]


@task
async def make_preds(x: pd.DataFrame):
    model = joblib.load("./models/rf_aapl_model.pkl")
    pred = model.predict(x)
    return pred[0]

@task
async def make_trades(signal: int):
    api_key = await String.load("alpaca-api-key")
    secret_key = await Secret.load("alpaca-secret-key")
    position = 0
    trading_client = TradingClient(api_key.value, secret_key.get(), paper=True)#, url_override="http://paper-api.alpaca.markets")
    current_positions = trading_client.get_all_positions()
    if len(current_positions) == 1:
        position = 1

    if position == 0 and signal == 1:
        # preparing orders
        market_order_data = MarketOrderRequest(
                            symbol="AAPL",
                            qty=100,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY
                            )

        # Market order
        trading_client.submit_order(
                        order_data=market_order_data
                    )

    if position == 1 and signal == -1:
        trading_client.close_all_positions(cancel_orders=True)
        
    

@flow(name="AAPL Trading Flow")
async def execute_aapl_trades():
    data = await pull_pricing_data()
    features = await create_features(data)
    signal = await make_preds(features)
    await make_trades(signal)
    


if __name__ == "__main__":
    asyncio.run(execute_aapl_trades())