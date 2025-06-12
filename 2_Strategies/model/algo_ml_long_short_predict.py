import backtrader as bt
from algo_base import *
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import load_model

class MyStrategy(StrategyTemplate):

    def __init__(self):
        super(MyStrategy, self).__init__()
        self.config["long_threshold"]=float(self.config["long_threshold"])
        self.config["short_threshold"]=float(self.config["short_threshold"])
        self.config["size"]=int(self.config["size"])
        self.config["profit_target_pct"]=float(self.config["profit_target_pct"])
        self.config["stop_target_pct"]=float(self.config["stop_target_pct"])

        self.order=None
        self.orderPlaced=False
                                
        self.model = load_model('model_long_short_predict.h5')
        
        # input / indicators
        self.repeatCount=15
        self.repeatStep=1
        
        self.profitTarget=self.config["profit_target_pct"]/100.0
        self.stopTarget=self.config["stop_target_pct"]/100.0
        self.size=self.config["size"]
         
        self.sma=[]
        self.roc=[]
        
        self.hData=["dt"]
        self.hData.append("close") 
        for a in range(0,self.repeatCount):
            tp=(a+1)*self.repeatStep+1
            self.hData.append("sma"+str(tp))
            self.sma.append(bt.talib.SMA(self.data, timeperiod=tp, plot=False))
        for a in range(0,self.repeatCount):
            tp=(a+1)*self.repeatStep+1
            self.hData.append("roc"+str(tp))
            self.roc.append(bt.talib.ROC(self.data, timeperiod=tp, plot=False))

    def init_broker(broker):
        broker.setcash(100000.0)
        broker.setcommission(commission=0.0) 
        
    def add_data(cerebro):
        data = btfeeds.GenericCSVData(
            dataname=MyStrategy.TRAIN_FILE,
            dtformat=('%Y-%m-%d'),
            timeframe=bt.TimeFrame.Days,
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

    def next(self):
        super(MyStrategy, self).next()
        
        dt=self.datas[0].datetime.datetime(0)
        cl=self.dataclose[0]
        inputRec=[]                

        #open
        inputRec0=[]
        inputRec0.append(cl)

        #sma
        for a in range(0,self.repeatCount):
            if math.isnan(self.sma[a][0]):
                inputRec0.append(cl)
            else:
                inputRec0.append(self.sma[a][0])

        m1=min(inputRec0)
        m2=max(inputRec0)
        for a in inputRec0:
            if m2-m1==0:
                inputRec.append(0)
            else:
                inputRec.append((a-m1)/(m2-m1))

        #roc
        for a in range(0,self.repeatCount):
            if math.isnan(self.roc[a][0]):
                inputRec.append(0)
            else:
                inputRec.append(self.roc[a][0])

        mX=[]
        dataX=np.array(mX)
        #print("dataX=%s" % dataX)

        # *** ML prediction ***
        mX=[]
        mX.append(np.array(inputRec))
        dataX=np.atleast_2d(np.array(mX)) # Ensure dataX is at least 2D        
        dataX_reshaped = dataX.reshape(dataX.shape[0], 1, dataX.shape[1])
        mY=self.model.predict(dataX_reshaped)
        mY_squeezed = np.squeeze(mY)
        #print("mY=%s" % mY)
        tLong = mY_squeezed[0] # mY_squeezed will be a 1D array like [val1, val2]
        tShort = mY_squeezed[1]
        #tLong = mY[0][0][0]  # Accesses the first element of the innermost array
        #tShort = mY[0][0][1] # Accesses the second element of the innermost array  
        #print("[%s]:long=%s,short=%s" % (dt,tLong,tShort))
        if not self.position:
            fLong=(tLong>self.config["long_threshold"]) 
            fShort=(tShort>self.config["short_threshold"])
            if fLong:
                self.order=self.buy(size=self.size)
                self.limitPrice=cl+self.profitTarget*cl
                self.stopPrice=cl-self.stopTarget*cl
            elif fShort:
                self.order=self.sell(size=self.size)                    
                self.limitPrice=cl-self.profitTarget*cl
                self.stopPrice=cl+self.stopTarget*cl

        if self.position:
            if self.position.size>0:
                if cl>=self.limitPrice or cl<=self.stopPrice:
                    self.order=self.sell(size=self.size)
            elif self.position.size<0:
                if cl<=self.limitPrice or cl>=self.stopPrice:
                    self.order=self.buy(size=self.size)
