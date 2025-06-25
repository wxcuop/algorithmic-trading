import backtrader as bt
from algo_base import *

class MyStrategy(StrategyTemplate):

    def __init__(self):  # Initiation
        super(MyStrategy, self).__init__()
        self.config["fast_period"]=int(self.config["fast_period"])
        self.config["slow_period"]=int(self.config["slow_period"])
        self.config["size"]=int(self.config["size"])

        # Calculate indicators using Backtrader
        self.smaFast = bt.ind.SimpleMovingAverage(period=self.config["fast_period"])
        self.smaSlow = bt.ind.SimpleMovingAverage(period=self.config["slow_period"])
        self.size = self.config["size"]

    # Set cash and commissions
    def init_broker(broker):
        broker.setcash(100000.0)
        broker.setcommission(commission=0.0) 
        
    # Cerebro is an algo engine
    def add_data(cerebro):
        data = btfeeds.GenericCSVData(
            dataname=MyStrategy.TRAIN_FILE,
            dtformat=('%Y-%m-%d'),
            timeframe=bt.TimeFrame.Days,
            # These are just indicies of columns in the file
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

    # Processnext data point (in our case a daily bar
    def next(self):  # Processing
        super(MyStrategy, self).next()
        dt=self.datas[0].datetime.datetime(0)
        if not self.position:
            if self.smaFast[0] > self.smaSlow[0]:
                self.buy(size=self.size) # Go long
            else:
                self.sell(size=self.size) # Go short
        elif self.position.size>0 and self.smaFast[0] < self.smaSlow[0]:
            self.sell(size=2*self.size) # Go short
        elif self.position.size<0 and self.smaFast[0] > self.smaSlow[0]:          
            self.buy(size=2*self.size) # Go long
