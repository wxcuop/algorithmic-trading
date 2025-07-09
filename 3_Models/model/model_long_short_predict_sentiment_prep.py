#!/usr/bin/env python
# This line is a 'shebang', telling the system to execute the script using the python interpreter.

import numpy as np 
import pandas as pd
import talib as ta  # Import TA-Lib (Technical Analysis Library) for financial indicators.
from talib.abstract import * # Import all functions from talib.abstract directly for easier use (e.g., SMA, ROC).
import math

prefix = '/opt/ml/'
input_path = prefix + 'input/data/training'

data_orig_file = input_path + '/data_orig.csv'
data_file = input_path + '/data.csv'

# --- Load Original Data ---
# Read the original CSV file into a Pandas DataFrame.
d = pd.read_csv(data_orig_file, parse_dates=['dt'], index_col=['dt'])
print(d.head()) # Print the first few rows of the loaded DataFrame for inspection.

# --- Define Parameters for Indicator Calculation and Signal Generation ---
repeatCount = 15  # How many different periods of SMA and ROC indicators will be calculated.
repeatStep = 1    # The increment for the periods of the indicators (e.g., period 2, then 3, then 4...).
lookBack = repeatCount * repeatStep # Calculates a 'look back' period, related to max indicator period.
forwardWindow = 5 # IMPORTANT - the number of future data points to look at when determining "long" and "short" signals.
                  # This defines the future window for profit/stop-loss checks.

profitTarget = 2.0 / 100.0 # Desired profit percentage (2%) as a decimal to trigger a "long" or "short" signal.
stopTarget = 1.5 / 100.0   # Maximum acceptable loss percentage (1.5%) as a decimal before a signal is invalidated.

iCount = lookBack # A counter initialized to lookBack (not explicitly used as a loop condition later, but set).
                  # Its purpose is usually to skip initial rows where technical indicators might not have enough data to be calculated.

hData = ["dt"]
hData.append("close")

# Add column names for the Simple Moving Average (SMA) indicators.
# Periods are (a+2)*repeatStep, e.g., for repeatStep=1, periods are 2, 3, 4, ..., 16.
for a in range(0, repeatCount):
    hData.append("sma" + str((a + 2) * repeatStep))

# Add column names for the Rate of Change (ROC) indicators.
# Periods are (a+2)*repeatStep, e.g., for repeatStep=1, periods are 2, 3, 4, ..., 16.
for a in range(0, repeatCount):
    hData.append("roc" + str((a + 2) * repeatStep))

# Add column names for the Sentiment indicators.
# Periods are (a+2)*repeatStep, e.g., for repeatStep=1, periods are 2, 3, 4, ..., 16.
for a in range(0, repeatCount):
    hData.append("sentiment_score" + str((a + 2) * repeatStep))

hData.append("long")  # Add column for the 'long' binary signal (0 or 1).
hData.append("short") # Add column for the 'short' binary signal (0 or 1).

# --- Initialize List to Store Processed Data Rows ---
tData = [] # This list will accumulate all the rows for the new DataFrame.

# --- Prepare Input for TA-Lib Calculations ---
# TA-Lib functions typically expect NumPy arrays as input.
inputs = {
    'close': np.array(d["close"]), # Extract the 'close' column as a NumPy array.
    'sentiment_score': np.array(d["sentiment_score"]) # Extract the 'sentiment_score' column as a NumPy array.
}

# --- Calculate SMA (Simple Moving Average) Indicators ---
sma = [] 
# This is where talib is used - to calculate technical indicators.
for a in range(0, repeatCount):
    # Calculate SMA for periods (a+1)*repeatStep+1 (e.g., 2, 3, ..., 16 if repeatStep=1)
    sma.append(SMA(inputs['close'], timeperiod=(a + 1) * repeatStep + 1))

# --- Calculate ROC (Rate of Change) Indicators ---
roc = [] 
for a in range(0, repeatCount):
    # Calculate ROC for periods (a+1)*repeatStep+1
    roc.append(ROC(inputs['close'], timeperiod=(a + 1) * repeatStep + 1))

sentiment_score = [] 

for a in range(0, repeatCount):
    # Calculate Sentiment SMA for periods (a+1)*repeatStep+1 (e.g., 2, 3, ..., 16 if repeatStep=1)
    sentiment_score.append(SMA(inputs['sentiment_score'], timeperiod=(a + 1) * repeatStep + 1))

# --- Prepare for Iteration and Counters ---
closeList = d["close"] # Get the 'close' column as a Pandas Series for easy access to future prices.
sentimentList = d["sentiment_score"] # Get the 'close' column as a Pandas Series for easy access to future prices.

dLen = len(d)          # Get the total number of rows in the original DataFrame.
n = 0                  # Row counter for iterating through the DataFrame.
lCount = 0             # Counter for 'long' signals generated.
sCount = 0             # Counter for 'short' signals generated.
nCount = 0             # Unused counter.
n = 0                  # Resetting 'n' (redundant if declared right before the loop).

# --- Main Loop: Process Each Row of Data ---
# Iterate through each row (index and row data) of the original DataFrame.
for idx, row in d.iterrows():
    # Ensure there are enough future data points (defined by forwardWindow)
    # to calculate the long/short signals without going out of bounds.
    if n < dLen - forwardWindow - 1:
        dt1 = idx          # Current row's datetime.
        cl = row["close"]  # Current row's closing price.
        st = row["sentiment_score"]  # Current row's closing price.        
        inputRec = []      # List to store the features for the current row of the output DataFrame.
        inputRec.append(idx) # Add the datetime to the output record.

        inputRec0 = [] # Temporary list to hold the current close price and SMA values for normalization.

        # --- Add Current Close Price to Temporary List ---
        inputRec0.append(cl)

        # --- Process SMA Values ---
        for a in range(0, repeatCount):
            # If an SMA value is NaN (Not a Number, usually at the beginning of the series because
            # there isn't enough historical data to calculate the indicator),
            # it appends the current closing price instead as a fallback.
            if math.isnan(sma[a][n]):
                inputRec0.append(cl)
            else:
                inputRec0.append(sma[a][n]) # Otherwise, append the calculated SMA value.

        # --- Min-Max Normalization for Close Price and SMA Values ---
        # Scale these values to be between 0 and 1. This is a common preprocessing step for ML models.
        m1 = min(inputRec0) # Find the minimum value in the temporary list.
        m2 = max(inputRec0) # Find the maximum value in the temporary list.
        for val in inputRec0:
            if m2 - m1 == 0: # If min and max are the same (e.g., all values are identical), avoid division by zero.
                inputRec.append(0) # Append 0 in this case.
            else:
                # Apply the Min-Max Normalization formula: (value - min) / (max - min)
                inputRec.append((val - m1) / (m2 - m1))

        # --- Process ROC Values ---
        # Note: ROC values are not normalized in this section, unlike SMA/close.
        for a in range(0, repeatCount):
            if math.isnan(roc[a][n]): # Check if the ROC value is NaN.
                inputRec.append(0)    # If NaN, append 0 as a fallback.
            else:
                inputRec.append(roc[a][n]) # Otherwise, append the calculated ROC value.

        # --- Process Sentiment Values ---
        # Note: Sentiment values are not normalized in this section, unlike SMA/close.
        for a in range(0, repeatCount):
            if math.isnan(sentiment_score[a][n]): # Check if the ROC value is NaN.
                inputRec.append(0)    # If NaN, append 0 as a fallback.
            else:
                inputRec.append(sentiment_score[a][n]) # Otherwise, append the calculated Sentiment value.
        
        # --- Analyze Future Prices for Signal Generation ---
        # Gets a list of future closing prices within the 'forwardWindow' (e.g., next 5 days).
        # 'min(dLen-1, n+1+forwardWindow)' prevents going out of bounds at the end of the data.
        rClose = closeList[n + 1:min(dLen - 1, n + 1 + forwardWindow)].values.tolist()
        low = min(rClose)  # Find the minimum price in the future window.
        high = max(rClose) # Find the maximum price in the future window.

        # --- Determine 'Long' Signal ---
        long = 0 # Initialize 'long' signal to 0 (no signal).
        # This is the core logic for a "long" signal:
        # Condition 1: Future high must reach or exceed current close + profit target.
        # Condition 2: Future low must stay above or equal to current close - stop target (prevent significant drops).
        if high >= cl + cl * profitTarget and low >= cl - cl * stopTarget:
            long = 1 # Set 'long' signal to 1.
            lCount = lCount + 1 # Increment the 'long' signal counter.
        inputRec.append(long) # Add the 'long' signal to the current row's features.

        # --- Determine 'Short' Signal ---
        short = 0 # Initialize 'short' signal to 0 (no signal).
        # This is the core logic for a "short" signal:
        # Condition 1: Future low must reach or go below current close - profit target.
        # Condition 2: Future high must stay below or equal to current close + stop target (prevent significant rises).
        if low <= cl - cl * profitTarget and high <= cl + cl * stopTarget:
            short = 1 # Set 'short' signal to 1.
            sCount = sCount + 1 # Increment the 'short' signal counter.
        inputRec.append(short) # Add the 'short' signal to the current row's features.

        tData.append(inputRec) # Add the complete processed row to the list of all data.
        n = n + 1 # Move to the next row.

# Print the total counts of 'long' and 'short' signals generated.
print("lCount=%s,sCount=%s" % (lCount, sCount))

df1 = pd.DataFrame(tData, columns=hData)
df1.set_index(pd.DatetimeIndex(df1['dt']), inplace=True)
del df1['dt'] 
df1.to_csv(data_file)
print(df1.head()) 
print("count=%s" % (len(df1)))
