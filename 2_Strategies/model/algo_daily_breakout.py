import backtrader as bt
from algo_base import *
import pytz
from pytz import timezone

class MyStrategy(StrategyTemplate):

    def __init__(self):  # Initiation
        super(MyStrategy, self).__init__()
        self.config["period"]=int(self.config["period"])
        self.config["size"]=int(self.config["size"])
        self.config["go_long"]=(str(self.config["go_long"]).lower()=="true")
        self.config["go_short"]=(str(self.config["go_short"]).lower()=="true")

        self.highest = bt.ind.Highest(period=self.config["period"])
        self.lowest = bt.ind.Lowest(period=self.config["period"])
        self.size = self.config["size"]
        
    # Set the cash and commissions
    def init_broker(broker):
        broker.setcash(100000.0)
        broker.setcommission(commission=0.0) 
        
    # Cerebro is the algo engine of the backtesting library. We add market data here
    def add_data(cerebro):
        data = btfeeds.GenericCSVData(
            dataname=MyStrategy.TRAIN_FILE,
            dtformat=('%Y-%m-%d'),
            timeframe=bt.TimeFrame.Days,
            # These are just indecies of the columns in the file
            datetime=0,
            time=-1,
            high=2,
            low=3,
            open=1,
            close=4,
            volume=5,
            openinterest=-1
        )
        cerebro.adddata(data)

    # Process next data (here we deal with daily bars)
    def next(self):  # Processing
        super(MyStrategy, self).next()
        dt=self.datas[0].datetime.datetime(0)
        if not self.position:
            # If 'go_long' is enabled and the current close price is greater than the previous highest high
            if self.config["go_long"] and self.datas[0] > self.highest[-1]:
                self.buy(size=self.size) # Go long
            # Else if 'go_short' is enabled and the current close price is less than the previous lowest low              
            elif self.config["go_short"] and self.datas[0] < self.lowest[-1]:
                self.sell(size=self.size) # Go short
        # If there is an open long position (size > 0) and the current close price drops below the previous highest high               
        elif self.position.size>0 and self.datas[0] < self.highest[-1]:
            self.close() # Close the long position
        # If there is an open short position (size < 0) and the current close price rises above the previous lowest low           
        elif self.position.size<0 and self.datas[0] > self.lowest[-1]:          
            self.close()  # Close the short position
