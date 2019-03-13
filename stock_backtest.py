import subprocess
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


class BackTester:
    """
    Class to represent the stock back testing tool
    """
    def __init__(self,
                 
                 start_date=datetime(2010, 1, 1),
                 token="05jZzQcNaNnqkHnobe0X2txoIKWASFaysMLN3OGNpQMCfyf1BdJ3xJdGHcgv",
                 x1=20,
                 x2=55
                 ):
        """
        Constructor
        :param start_date: start date to collect stock data from
        :param token: worldtradingdata token
        :param x1: time horizon 1 for trades
        :param x2: time horizon 2 for trades
        """
        self.token = token
        self.tickers = {"ftse100": np.genfromtxt('./data/ftse100.csv', dtype=str, delimiter=',', skip_header=1)}
        self.stocks = {}
        self.start_date = start_date
        self.x1 = x1
        self.x2 = x2

    def pickle_data(self, ticker, pickle_folder):
        """
        method to pickle stock data
        :param ticker: stock ticker
        :param pickle_folder: folder to save to
        :return: 
        """
        url = f"https://www.worldtradingdata.com/api/v1/history?symbol={ticker}&sort=newest&api_token={self.token}"
        data = subprocess.Popen(['curl', '-s', url], stdout=subprocess.PIPE).communicate()[0]
        data = pd.DataFrame.from_dict(json.loads(data.decode('utf-8'))["history"], orient='index')
        pickle.dump(data, open(f"{pickle_folder}/{ticker}.p", "wb"))

    def add_data(self, ticker):
        """
        method to add stock data
        :param ticker: stock ticker
        :return: 
        """
        url = f"https://www.worldtradingdata.com/api/v1/history?symbol={ticker}&sort=newest&api_token={self.token}"
        data = subprocess.Popen(['curl', '-s', url], stdout=subprocess.PIPE).communicate()[0]
        data = pd.DataFrame.from_dict(json.loads(data.decode('utf-8'))["history"], orient='index')
        self.stocks[ticker] = Stock(ticker, data)

    def calc_metrics(self, ticker):
        """
        method to calculate statistics on stock price history
        :param ticker: stock ticker
        :return: 
        """
        self.stocks[ticker].calc_metrics(self.start_date, self.x1, self.x2)

    def back_test(self, ticker):
        """
        Method to generate trades based on historical prices
        :param ticker: stock ticker
        :return: 
        """
        for index, row in self.stocks[ticker].data.iterrows():
            self.stocks[ticker].turtle_trade(row)

    def plot(self, ticker):
        """
        Merthod to plot strategy performance against buy and hold strategy
        :param ticker: stock ticker
        :return: 
        """
        self.stocks[ticker].data["strategy_100000"] = self.stocks[ticker].data["strategy_100000"].fillna(method='ffill')
        self.stocks[ticker].data["buy_hold_100000"].plot(grid=True)
        self.stocks[ticker].data["strategy_100000"].plot(grid=True)
        plt.savefig(f"./plot/{ticker}.png", dpi=300)
        plt.close()

    def pickle_ftse100(self):
        """
        Method to pickle data for all FTSE100 stocks
        :return: 
        """
        for ticker in self.tickers["ftse100"]:
            self.pickle_data(ticker, './pickle/')

    def load_pickle(self, ticker):
        """
        load pickle data
        :param ticker: stock ticker
        :return: 
        """
        with open(f"./pickle/{ticker}.p", 'rb') as handle:
            self.stocks[ticker] = Stock(ticker, pickle.load(handle))


class Stock:
    """
    Class to represent a stock
    """
    def __init__(self, ticker, data):
        """
        Constructor
        :param ticker: stock ticker
        :param data: stock data
        """
        self.data = data
        self.ticker = ticker
        self.trades_open = []
        self.trades_closed = []
        self.trades_open_count = 0
        self.next_entry = 0
        self.buy_hold_return = None
        self.strategy_return = None
        self.strategy_running = 100000

    def calc_metrics(self, start_date, x1, x2):
        """
        method to calculate statistics from historical stock prices
        :param start_date: start date for stock data
        :param x1: horizon 1 for moving averages, days
        :param x2: horizon 2 for moving averages, days
        :return: 
        """
        self.data.index = pd.to_datetime(self.data.index)
        self.data = self.data.loc[lambda c: c.index >= start_date]
        self.data["date"] = self.data.index
        self.data.index = np.arange(0, len(self.data))
        self.data[["high", "low", "open", "close", "volume"]] = \
            self.data[["high", "low", "open", "close", "volume"]].apply(pd.to_numeric)
        self.data["PDC"] = self.data["close"].shift(1)
        self.data["TR"] = self.data.apply(lambda row: max(row.high - row.low, row.high - row.PDC, row.PDC - row.low),
                                          axis=1)

        self.data[f"high_{x1}d"] = self.data["high"].shift(1).rolling(min_periods=1, window=x1, center=False).max()
        self.data[f"low_{x1}d"] = self.data["low"].shift(1).rolling(min_periods=1, window=x1, center=False).min()
        self.data[f"high_{x2}d"] = self.data["high"].shift(1).rolling(min_periods=1, window=x2, center=False).max()
        self.data[f"low_{x2}d"] = self.data["low"].shift(1).rolling(min_periods=1, window=x2, center=False).min()
        self.data["TR_EMA"] = 0
        self.data["TR_EMA"].iloc[x1] = np.mean(self.data["TR"].head(x1))
        for i in range(x1 + 1, len(self.data)):
            self.data.loc[i, "TR_EMA"] = (self.data.loc[i - 1, "TR_EMA"] * (x1 - 1) + self.data.loc[i, "TR"]) / x1
        self.data["volatility"] = self.data["TR_EMA"] / self.data["close"]
        self.data["unit"] = 100 / self.data["volatility"]
        self.data["long_entry"] = self.data.close > self.data[f"high_{x2}d"]
        self.data["long_exit"] = self.data.close < self.data[f"low_{x1}d"]
        self.data['positions_long'] = np.nan
        self.data.loc[self.data.long_entry, 'positions_long'] = 1
        self.data.loc[self.data.long_exit, 'positions_long'] = 0
        self.data = self.data.fillna(method='ffill')
        daily_log_returns = np.log(self.data.close / self.data.close.shift(1))
        self.data["daily_log_returns"] = daily_log_returns * self.data.positions_long.shift(1)
        self.data["daily_log_returns_cumsum"] = self.data.daily_log_returns.cumsum()
        self.data.drop(self.data.index[:x1], inplace=True)
        self.data["buy_hold_100000"] = self.data["close"] / self.data.loc[x1, "open"] * 100000
        self.data["strategy_100000"] = np.nan
        self.data.loc[x1, "strategy_100000"] = 100000
        self.data.index = self.data["date"]
        # self.data.set_index('date', inplace=True)

    def buy(self, date, price, units):
        """
        method to buy stock
        :param date: date of purchase
        :param price: price at purchase
        :param units: number of units
        :return: 
        """
        self.trades_open.append(Trade(self.ticker, date, price, units))

    def sell_all_open_trades(self, sell_price, date):
        """
        method to sell all open trades
        :param sell_price: sale price
        :param date: sale date
        :return: 
        """
        while self.trades_open:
            trade = self.trades_open.pop()
            trade.sell_date = date
            trade.sell_price = sell_price
            trade.sell_value = trade.units * sell_price
            trade.pl = trade.units * (trade.sell_price - trade.buy_price)
            trade.pl_pc = trade.pl / trade.buy_value
            self.trades_closed.append(trade)
            self.strategy_running += trade.pl
            self.data.loc[date, "strategy_100000"] = self.strategy_running

    def turtle_trade(self, row):
        """
        Method to execute the turtle trading strategy
        :param row: row of one days stock data
        :return: 
        """
        if self.trades_open_count > 0:
            if row.low < row.low_20d:
                self.sell_all_open_trades(row.low_20d, row.date)
                self.trades_open_count = 0
            elif row.high > self.next_entry:
                for t in np.arange(row.high_55d, row.high, row.TR_EMA / 2):
                    if self.trades_open_count < 4 and t > self.next_entry:
                        self.buy(row.date, max(t, row.open), int(25000 / max(t, row.open)))
                        self.trades_open_count += 1
                        self.next_entry = t + row.TR_EMA / 2
        else:
            if row.high > row.high_55d:
                for t in np.arange(row.high_55d, row.high, row.TR_EMA / 2):
                    if self.trades_open_count < 4:
                        self.buy(row.date, max(row.open, t), int(25000 / max(t, row.open)))
                        self.trades_open_count += 1
                        self.next_entry = t + row.TR_EMA / 2


class Trade:
    """
    Class to represent a trade
    """
    def __init__(self, ticker, date, price, units):
        """
        constructor
        :param ticker: stock ticker
        :param date: date of purchase
        :param price: price at purchase
        :param units: number of units
        """
        self.ticker = ticker
        self.buy_date = date
        self.units = units
        self.buy_price = price
        self.buy_value = units * price
        self.sell_date = None
        self.sell_price = None
        self.sell_value = None
        self.pl = None
        self.pl_pc = None


if __name__ == "__main__":
    bt = BackTester(start_date=datetime(2010, 1, 1),
                    x1=20,
                    x2=55)
    for ticker in bt.tickers["ftse100"]:
        bt.load_pickle(ticker)
        bt.calc_metrics(ticker)
        bt.back_test(ticker)
        bt.plot(ticker)
