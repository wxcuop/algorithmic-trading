#!/usr/bin/env python

import numpy as np
import pandas as pd
import talib as ta
from talib.abstract import *
import math

prefix = '/opt/ml/'
input_path = prefix + 'input/data/training'

data_orig_file = input_path+'/data_orig.csv'
data_file = input_path+'/data.csv'

d = pd.read_csv(data_orig_file, parse_dates=['dt'], index_col=['dt'])
print(d.head())

repeatCount=15
repeatStep=1
lookBack=repeatCount*repeatStep
forwardWindow=5

profitTarget=2.0/100.0
stopTarget=1.5/100.0

iCount=lookBack

# header
hData=["dt"]
hData.append("close")
for a in range(0,repeatCount):
    hData.append("sma"+str((a+2)*repeatStep))
for a in range(0,repeatCount):
    hData.append("roc"+str((a+2)*repeatStep))
hData.append("long")
hData.append("short")

# data
tData=[]

inputs = {
    'close': np.array(d["close"])
}
sma=[]
for a in range(0,repeatCount):
    sma.append(SMA(inputs,timeperiod=(a+1)*repeatStep+1))
roc=[]
for a in range(0,repeatCount):
    roc.append(ROC(inputs,timeperiod=(a+1)*repeatStep+1))

closeList=d["close"]
dLen=len(d)
n=0
lCount=0
sCount=0
nCount=0
n=0
for idx,row in d.iterrows():
    if n<dLen-forwardWindow-1:
        dt1=idx
        cl=row["close"]
        inputRec=[]
        inputRec.append(idx)

        inputRec0=[]

        #close
        inputRec0.append(cl)

        #sma
        for a in range(0,repeatCount):
            if math.isnan(sma[a][n]):
                inputRec0.append(cl)
            else:
                inputRec0.append(sma[a][n])

        m1=min(inputRec0)
        m2=max(inputRec0)
        for a in inputRec0:
            if m2-m1==0:
                inputRec.append(0)
            else:
                inputRec.append((a-m1)/(m2-m1))

        #roc
        for a in range(0,repeatCount):
            if math.isnan(roc[a][n]):
                inputRec.append(0)
            else:
                inputRec.append(roc[a][n])

        rClose=closeList[n+1:min(dLen-1,n+1+forwardWindow)].values.tolist()
        low=min(rClose)
        high=max(rClose)
        
        #long
        long=0
        if high>=cl+cl*profitTarget and low>=cl-cl*stopTarget:
            long=1
            lCount=lCount+1
        inputRec.append(long)
 
        #short
        short=0
        if low<=cl-cl*profitTarget and high<=cl+cl*stopTarget:
            short=1
            sCount=sCount+1
        inputRec.append(short)

        tData.append(inputRec)
        n=n+1
          
print("lCount=%s,sCount=%s" % (lCount,sCount))
df1=pd.DataFrame(tData,columns=hData)
df1.set_index(pd.DatetimeIndex(df1['dt']), inplace=True)
del df1['dt']
 
df1.to_csv(data_file)
print(df1.head())
print("count=%s" % (len(df1)))
