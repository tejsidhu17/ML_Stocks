import pandas as pd
import yfinance as yf
import numpy as np
from curl_cffi import requests

def get_features(tickers):
    ticker_data = []
    data_columns = ["Returns", "Log Returns", "Day Return", "High-Low Range", "Closing Strength",
                    "SMA_5", "SMA_10", "SMA_20", "SMA_50", "EMA_5", "EMA_10", "EMA_20", "EMA_50",
                    "RV_MA_5", "RV_MA_10", "RV_MA_20", "V_change", "Vol_5", "Vol_10", "Vol_30",
                    "Momentum_10", "Momentum_20", "Momentum_30", "MACD", "MACD_Signal", "MACD_Hist",
                    "MACD_Z", "MACD_Signal_Z", "MACD_Hist_Z", "Next_Day_Return"]
    for tick in tickers:
        print(f"Ticker: {tick}")
        try:
            ticker = yf.download(tick, period="10y", progress=False)
            ticker["Returns"] = ticker["Close"].pct_change()
            ticker["Log Returns"] = np.log(ticker["Close"]/ticker["Close"].shift(1))
            ticker["Day Return"] = (ticker["Close"] - ticker["Open"]) / ticker["Open"]
            ticker["High-Low Range"] = (ticker["High"] - ticker["Low"])/ticker["Close"]
            ticker["Closing Strength"] = (ticker["Close"] - ticker["Low"])/(ticker["High"] - ticker["Low"])

            ticker["SMA_5"] = (ticker["Close"]/(ticker["Close"].rolling(5).mean())) - 1
            ticker["SMA_10"] = (ticker["Close"]/(ticker["Close"].rolling(10).mean())) - 1
            ticker["SMA_20"] = (ticker["Close"]/(ticker["Close"].rolling(20).mean())) - 1
            ticker["SMA_50"] = (ticker["Close"]/(ticker["Close"].rolling(50).mean())) - 1
            ticker["EMA_5"] = (ticker["Close"]/(ticker["Close"].ewm(span=5, adjust=False).mean())) - 1
            ticker["EMA_10"] = (ticker["Close"]/(ticker["Close"].ewm(span=10, adjust=False).mean())) - 1
            ticker["EMA_20"] = (ticker["Close"]/(ticker["Close"].ewm(span=20, adjust=False).mean())) - 1
            ticker["EMA_50"] = (ticker["Close"]/(ticker["Close"].ewm(span=50, adjust=False).mean())) - 1

            ticker["RV_MA_5"] = ticker["Volume"]/(ticker["Volume"].rolling(5).mean())
            ticker["RV_MA_10"] = ticker["Volume"]/(ticker["Volume"].rolling(10).mean())
            ticker["RV_MA_20"] = ticker["Volume"]/(ticker["Volume"].rolling(20).mean())
            ticker["V_change"] = ticker["Volume"].pct_change()

            ticker["Vol_5"] = ticker["Returns"].rolling(5).std()
            ticker["Vol_10"] = ticker["Returns"].rolling(10).std()
            ticker["Vol_30"] = ticker["Returns"].rolling(30).std()

            ticker["Momentum_10"] = ticker["Close"].pct_change(10)
            ticker["Momentum_20"] = ticker["Close"].pct_change(20)
            ticker["Momentum_30"] = ticker["Close"].pct_change(30)

            ema_12 = ticker["Close"].ewm(span=12, adjust=False).mean()
            ema_26 = ticker["Close"].ewm(span=26, adjust=False).mean()
            ticker["MACD"] = ema_12 - ema_26
            ticker["MACD_Signal"] = ticker["MACD"].ewm(span=9, adjust=False).mean()
            ticker["MACD_Hist"] = ticker["MACD"] - ticker["MACD_Signal"]

            ticker["MACD_Z"] = (ticker["MACD"] - ticker["MACD"].rolling(20).mean()) / ticker["MACD"].rolling(20).std()
            ticker["MACD_Signal_Z"] = (ticker["MACD_Signal"] - ticker["MACD_Signal"].rolling(20).mean()) / ticker["MACD_Signal"].rolling(20).std()
            ticker["MACD_Hist_Z"] = (ticker["MACD_Hist"] - ticker["MACD_Hist"].rolling(20).mean()) / ticker["MACD_Hist"].rolling(20).std()

            ticker["Next_Day_Return"] = ticker["Returns"].shift(-1)
            ticker.replace([np.inf, -np.inf], np.nan, inplace=True)
            ticker.dropna(inplace=True)
            drop_columns = [col for col in ticker.columns if col[0] not in data_columns]
            ticker = ticker.drop(columns=drop_columns)
            ticker_data.append(ticker)
        except Exception as e:
            print(f"Exception occurred {e} with ticker: {ticker}")

    dataset = pd.concat(ticker_data, axis=0).reset_index(drop=True)
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.dropna(inplace=True)
    dataset.to_csv("stock_trading_indicators.csv", index=False)

def get_features_single(tick, period):
    session = requests.Session(impersonate="chrome")
    data_columns = ["Returns", "Log Returns", "Day Return", "High-Low Range", "Closing Strength",
                    "SMA_5", "SMA_10", "SMA_20", "SMA_50", "EMA_5", "EMA_10", "EMA_20", "EMA_50",
                    "RV_MA_5", "RV_MA_10", "RV_MA_20", "V_change", "Vol_5", "Vol_10", "Vol_30",
                    "Momentum_10", "Momentum_20", "Momentum_30", "MACD", "MACD_Signal", "MACD_Hist",
                    "MACD_Z", "MACD_Signal_Z", "MACD_Hist_Z", "Next_Day_Return"]
    print(f"Ticker: {tick}")
    ticker = yf.download(tick, period=period, session=session)
    ticker["Returns"] = ticker["Close"].pct_change()
    ticker["Log Returns"] = np.log(ticker["Close"] / ticker["Close"].shift(1))
    ticker["Day Return"] = (ticker["Close"] - ticker["Open"]) / ticker["Open"]
    ticker["High-Low Range"] = (ticker["High"] - ticker["Low"])/ticker["Close"]
    ticker["Closing Strength"] = (ticker["Close"] - ticker["Low"])/(ticker["High"] - ticker["Low"])

    ticker["SMA_5"] = (ticker["Close"]/(ticker["Close"].rolling(5).mean())) - 1
    ticker["SMA_10"] = (ticker["Close"]/(ticker["Close"].rolling(10).mean())) - 1
    ticker["SMA_20"] = (ticker["Close"]/(ticker["Close"].rolling(20).mean())) - 1
    ticker["SMA_50"] = (ticker["Close"]/(ticker["Close"].rolling(50).mean())) - 1
    ticker["EMA_5"] = (ticker["Close"]/(ticker["Close"].ewm(span=5, adjust=False).mean())) - 1
    ticker["EMA_10"] = (ticker["Close"]/(ticker["Close"].ewm(span=10, adjust=False).mean())) - 1
    ticker["EMA_20"] = (ticker["Close"]/(ticker["Close"].ewm(span=20, adjust=False).mean())) - 1
    ticker["EMA_50"] = (ticker["Close"]/(ticker["Close"].ewm(span=50, adjust=False).mean())) - 1

    ticker["RV_MA_5"] = ticker["Volume"]/(ticker["Volume"].rolling(5).mean())
    ticker["RV_MA_10"] = ticker["Volume"]/(ticker["Volume"].rolling(10).mean())
    ticker["RV_MA_20"] = ticker["Volume"]/(ticker["Volume"].rolling(20).mean())
    ticker["V_change"] = ticker["Volume"].pct_change()

    ticker["Vol_5"] = ticker["Returns"].rolling(5).std()
    ticker["Vol_10"] = ticker["Returns"].rolling(10).std()
    ticker["Vol_30"] = ticker["Returns"].rolling(30).std()

    ticker["Momentum_10"] = ticker["Close"].pct_change(10)
    ticker["Momentum_20"] = ticker["Close"].pct_change(20)
    ticker["Momentum_30"] = ticker["Close"].pct_change(30)

    ema_12 = ticker["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = ticker["Close"].ewm(span=26, adjust=False).mean()
    ticker["MACD"] = ema_12 - ema_26
    ticker["MACD_Signal"] = ticker["MACD"].ewm(span=9, adjust=False).mean()
    ticker["MACD_Hist"] = ticker["MACD"] - ticker["MACD_Signal"]

    ticker["MACD_Z"] = (ticker["MACD"] - ticker["MACD"].rolling(20).mean()) / ticker["MACD"].rolling(20).std()
    ticker["MACD_Signal_Z"] = (ticker["MACD_Signal"] - ticker["MACD_Signal"].rolling(20).mean()) / ticker["MACD_Signal"].rolling(20).std()
    ticker["MACD_Hist_Z"] = (ticker["MACD_Hist"] - ticker["MACD_Hist"].rolling(20).mean()) / ticker["MACD_Hist"].rolling(20).std()

    ticker["Next_Day_Return"] = ticker["Returns"].shift(-1)
    ticker.replace([np.inf, -np.inf], np.nan, inplace=True)
    ticker.dropna(inplace=True)
    drop_columns = [col for col in ticker.columns if col[0] not in data_columns]
    ticker = ticker.drop(columns=drop_columns)
    return ticker

def get_features_CNN(tickers):
    ticker_data = []
    data_columns = ["Returns", "Log Returns", "Day Return", "High-Low Range", "Closing Strength",
                    "SMA_5", "SMA_10", "SMA_20", "SMA_50", "EMA_5", "EMA_10", "EMA_20", "EMA_50",
                    "RV_MA_5", "RV_MA_10", "RV_MA_20", "V_change", "Vol_5", "Vol_10", "Vol_30",
                    "Momentum_10", "Momentum_20", "Momentum_30", "MACD", "MACD_Signal", "MACD_Hist",
                    "MACD_Z", "MACD_Signal_Z", "MACD_Hist_Z", "Next_Week_Return"]
    for tick in tickers:
        print(f"Ticker: {tick}")
        try:
            ticker = yf.download(tick, period="10y")
            ticker.index = pd.to_datetime(ticker.index)
            ticker.index.name = "Date"
            ticker["Returns"] = ticker["Close"].pct_change()
            ticker["Log Returns"] = np.log(ticker["Close"] / ticker["Close"].shift(1))
            ticker["Day Return"] = (ticker["Close"] - ticker["Open"]) / ticker["Open"]
            ticker["High-Low Range"] = (ticker["High"] - ticker["Low"])/ticker["Close"]
            ticker["Closing Strength"] = (ticker["Close"] - ticker["Low"])/(ticker["High"] - ticker["Low"])

            ticker["SMA_5"] = (ticker["Close"]/(ticker["Close"].rolling(5).mean())) - 1
            ticker["SMA_10"] = (ticker["Close"]/(ticker["Close"].rolling(10).mean())) - 1
            ticker["SMA_20"] = (ticker["Close"]/(ticker["Close"].rolling(20).mean())) - 1
            ticker["SMA_50"] = (ticker["Close"]/(ticker["Close"].rolling(50).mean())) - 1
            ticker["EMA_5"] = (ticker["Close"]/(ticker["Close"].ewm(span=5, adjust=False).mean())) - 1
            ticker["EMA_10"] = (ticker["Close"]/(ticker["Close"].ewm(span=10, adjust=False).mean())) - 1
            ticker["EMA_20"] = (ticker["Close"]/(ticker["Close"].ewm(span=20, adjust=False).mean())) - 1
            ticker["EMA_50"] = (ticker["Close"]/(ticker["Close"].ewm(span=50, adjust=False).mean())) - 1

            ticker["RV_MA_5"] = ticker["Volume"]/(ticker["Volume"].rolling(5).mean())
            ticker["RV_MA_10"] = ticker["Volume"]/(ticker["Volume"].rolling(10).mean())
            ticker["RV_MA_20"] = ticker["Volume"]/(ticker["Volume"].rolling(20).mean())
            ticker["V_change"] = ticker["Volume"].pct_change()

            ticker["Vol_5"] = ticker["Returns"].rolling(5).std()
            ticker["Vol_10"] = ticker["Returns"].rolling(10).std()
            ticker["Vol_30"] = ticker["Returns"].rolling(30).std()

            ticker["Momentum_10"] = ticker["Close"].pct_change(10)
            ticker["Momentum_20"] = ticker["Close"].pct_change(20)
            ticker["Momentum_30"] = ticker["Close"].pct_change(30)

            ema_12 = ticker["Close"].ewm(span=12, adjust=False).mean()
            ema_26 = ticker["Close"].ewm(span=26, adjust=False).mean()
            ticker["MACD"] = ema_12 - ema_26
            ticker["MACD_Signal"] = ticker["MACD"].ewm(span=9, adjust=False).mean()
            ticker["MACD_Hist"] = ticker["MACD"] - ticker["MACD_Signal"]

            ticker["MACD_Z"] = (ticker["MACD"] - ticker["MACD"].rolling(20).mean()) / ticker["MACD"].rolling(20).std()
            ticker["MACD_Signal_Z"] = (ticker["MACD_Signal"] - ticker["MACD_Signal"].rolling(20).mean()) / ticker["MACD_Signal"].rolling(20).std()
            ticker["MACD_Hist_Z"] = (ticker["MACD_Hist"] - ticker["MACD_Hist"].rolling(20).mean()) / ticker["MACD_Hist"].rolling(20).std()

            ticker["Next_Week_Return"] = (ticker["Close"].shift(-5) / ticker["Close"]) - 1
            ticker.replace([np.inf, -np.inf], np.nan, inplace=True)
            ticker.dropna(inplace=True)
            drop_columns = [col for col in ticker.columns if col[0] not in data_columns]
            ticker = ticker.drop(columns=drop_columns)
            ticker_data.append(ticker)
        except Exception as e:
            print(f"Exception occurred {e} with ticker: {ticker}")

    return ticker_data

def get_features_LSTM(tickers):
    ticker_data = []
    data_columns = ["Returns", "Log Returns", "Day Return", "High-Low Range", "Closing Strength",
                    "SMA_5", "SMA_10", "SMA_20", "SMA_50", "EMA_5", "EMA_10", "EMA_20", "EMA_50",
                    "RV_MA_5", "RV_MA_10", "RV_MA_20", "V_change", "Vol_5", "Vol_10", "Vol_30",
                    "Momentum_10", "Momentum_20", "Momentum_30", "MACD", "MACD_Signal", "MACD_Hist",
                    "MACD_Z", "MACD_Signal_Z", "MACD_Hist_Z", "Next_Day_Return"]
    for tick in tickers:
        print(f"Ticker: {tick}")
        try:
            ticker = yf.download(tick, period="10y", progress=False)
            ticker.index = pd.to_datetime(ticker.index)
            ticker.index.name = "Date"
            ticker["Returns"] = ticker["Close"].pct_change()
            ticker["Log Returns"] = np.log(ticker["Close"] / ticker["Close"].shift(1))
            ticker["Day Return"] = (ticker["Close"] - ticker["Open"]) / ticker["Open"]
            ticker["High-Low Range"] = (ticker["High"] - ticker["Low"])/ticker["Close"]
            ticker["Closing Strength"] = (ticker["Close"] - ticker["Low"])/(ticker["High"] - ticker["Low"])

            ticker["SMA_5"] = (ticker["Close"]/(ticker["Close"].rolling(5).mean())) - 1
            ticker["SMA_10"] = (ticker["Close"]/(ticker["Close"].rolling(10).mean())) - 1
            ticker["SMA_20"] = (ticker["Close"]/(ticker["Close"].rolling(20).mean())) - 1
            ticker["SMA_50"] = (ticker["Close"]/(ticker["Close"].rolling(50).mean())) - 1
            ticker["EMA_5"] = (ticker["Close"]/(ticker["Close"].ewm(span=5, adjust=False).mean())) - 1
            ticker["EMA_10"] = (ticker["Close"]/(ticker["Close"].ewm(span=10, adjust=False).mean())) - 1
            ticker["EMA_20"] = (ticker["Close"]/(ticker["Close"].ewm(span=20, adjust=False).mean())) - 1
            ticker["EMA_50"] = (ticker["Close"]/(ticker["Close"].ewm(span=50, adjust=False).mean())) - 1

            ticker["RV_MA_5"] = ticker["Volume"]/(ticker["Volume"].rolling(5).mean())
            ticker["RV_MA_10"] = ticker["Volume"]/(ticker["Volume"].rolling(10).mean())
            ticker["RV_MA_20"] = ticker["Volume"]/(ticker["Volume"].rolling(20).mean())
            ticker["V_change"] = ticker["Volume"].pct_change()

            ticker["Vol_5"] = ticker["Returns"].rolling(5).std()
            ticker["Vol_10"] = ticker["Returns"].rolling(10).std()
            ticker["Vol_30"] = ticker["Returns"].rolling(30).std()

            ticker["Momentum_10"] = ticker["Close"].pct_change(10)
            ticker["Momentum_20"] = ticker["Close"].pct_change(20)
            ticker["Momentum_30"] = ticker["Close"].pct_change(30)

            ema_12 = ticker["Close"].ewm(span=12, adjust=False).mean()
            ema_26 = ticker["Close"].ewm(span=26, adjust=False).mean()
            ticker["MACD"] = ema_12 - ema_26
            ticker["MACD_Signal"] = ticker["MACD"].ewm(span=9, adjust=False).mean()
            ticker["MACD_Hist"] = ticker["MACD"] - ticker["MACD_Signal"]

            ticker["MACD_Z"] = (ticker["MACD"] - ticker["MACD"].rolling(20).mean()) / ticker["MACD"].rolling(20).std()
            ticker["MACD_Signal_Z"] = (ticker["MACD_Signal"] - ticker["MACD_Signal"].rolling(20).mean()) / ticker["MACD_Signal"].rolling(20).std()
            ticker["MACD_Hist_Z"] = (ticker["MACD_Hist"] - ticker["MACD_Hist"].rolling(20).mean()) / ticker["MACD_Hist"].rolling(20).std()

            ticker["Next_Day_Return"] = ticker["Returns"].shift(-1)
            ticker.replace([np.inf, -np.inf], np.nan, inplace=True)
            ticker.dropna(inplace=True)
            drop_columns = [col for col in ticker.columns if col[0] not in data_columns]
            ticker = ticker.drop(columns=drop_columns)
            ticker_data.append(ticker)
        except Exception as e:
            print(f"Exception occurred {e} with ticker: {ticker}")

    return ticker_data
