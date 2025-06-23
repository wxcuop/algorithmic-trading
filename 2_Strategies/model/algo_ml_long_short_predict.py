import backtrader as bt  # Import the Backtrader framework for backtesting and trading strategies
from algo_base import * # Import custom base classes or utilities, likely including StrategyTemplate
import math  # Import the math module for mathematical operations (e.g., checking for NaN)
import numpy as np  # Import NumPy for numerical operations, especially array manipulation
import pandas as pd  # Import Pandas for data manipulation (though not directly used for dataframes in this class)
import tensorflow as tf  # Import TensorFlow, the deep learning framework
import keras  # Import Keras, a high-level API for building and training neural networks (now part of TensorFlow)
from keras import backend as K  # Import Keras backend for low-level operations (often used for clearing session)
from keras.models import load_model  # Import function to load a pre-trained Keras model

# Define the trading strategy class, inheriting from StrategyTemplate (a custom base for Backtrader)
class MyStrategy(StrategyTemplate):

    # This method is called once when the strategy is first initialized by Backtrader
    def __init__(self):
        # Call the constructor of the parent class (StrategyTemplate) to ensure its initialization
        super(MyStrategy, self).__init__()

        # --- Load Strategy Configuration Parameters ---
        # These parameters are typically passed into the strategy from an external configuration
        self.config["long_threshold"] = float(self.config["long_threshold"])   # ML prediction confidence needed to go long
        self.config["short_threshold"] = float(self.config["short_threshold"]) # ML prediction confidence needed to go short
        self.config["size"] = int(self.config["size"])                         # Number of shares/units to trade
        self.config["profit_target_pct"] = float(self.config["profit_target_pct"]) # Percentage profit target for exiting a trade
        self.config["stop_target_pct"] = float(self.config["stop_target_pct"])   # Percentage stop-loss target for exiting a trade

        # --- Initialize Internal Strategy Variables ---
        self.order = None      # Stores the current pending/active order (e.g., buy, sell)
        self.orderPlaced = False # Flag to track if an order has been recently placed (not explicitly used here but common)

        # --- Load the Pre-trained Machine Learning Model ---
        # This loads the neural network 'brain' that will provide buy/sell signals
        self.model = load_model('model_long_short_predict.h5')

        # --- Configuration for Technical Indicators ---
        self.repeatCount = 15  # Number of different periods for SMA and ROC calculations
        self.repeatStep = 1    # Step size for increasing the periods

        # --- Calculate Absolute Profit/Stop-Loss Targets from Percentages ---
        # Convert percentage targets to decimal values for calculation
        self.profitTarget = self.config["profit_target_pct"] / 100.0
        self.stopTarget = self.config["stop_target_pct"] / 100.0
        self.size = self.config["size"] # Store the trade size

        # --- Initialize Lists for Technical Indicators ---
        self.sma = []  # List to hold Backtrader's SMA indicator lines
        self.roc = []  # List to hold Backtrader's ROC indicator lines

        # --- Prepare Header for Data (if needed for debugging/export, though not used for model input directly) ---
        self.hData = ["dt"]
        self.hData.append("close")

        # --- Create SMA (Simple Moving Average) Indicators ---
        # Loop to create 'repeatCount' different SMA indicators, each with a different timeperiod
        for a in range(0, self.repeatCount):
            tp = (a + 1) * self.repeatStep + 1  # Calculate the time period for the current SMA (e.g., 2, 3, ..., 16)
            self.hData.append("sma" + str(tp))  # Add SMA column name to header
            # Add SMA indicator to the strategy, linked to the main data feed
            # plot=False prevents it from being plotted by default in Backtrader charts
            self.sma.append(bt.talib.SMA(self.data, timeperiod=tp, plot=False))

        # --- Create ROC (Rate of Change) Indicators ---
        # Loop to create 'repeatCount' different ROC indicators, each with a different timeperiod
        for a in range(0, self.repeatCount):
            tp = (a + 1) * self.repeatStep + 1  # Calculate the time period for the current ROC
            self.hData.append("roc" + str(tp))  # Add ROC column name to header
            # Add ROC indicator to the strategy, linked to the main data feed
            self.roc.append(bt.talib.ROC(self.data, timeperiod=tp, plot=False))

    # --- Static Method: Initializes the Trading Broker (Simulated Account) ---
    # This function is called by Cerebro to set up the trading environment
    def init_broker(broker):
        broker.setcash(100000.0)      # Set the initial cash in the simulated trading account
        broker.setcommission(commission=0.0) # Set trading commission to 0 (for simplicity/testing)

    # --- Static Method: Adds Market Data to Cerebro ---
    # This function defines how the historical market data is loaded
    def add_data(cerebro):
        data = btfeeds.GenericCSVData( # Use GenericCSVData to read data from a CSV file
            dataname=MyStrategy.TRAIN_FILE, # Path to the CSV file (TRAIN_FILE is likely defined elsewhere)
            dtformat=('%Y-%m-%d'),      # Date format in the CSV
            timeframe=bt.TimeFrame.Days, # Data is daily
            datetime=0,                 # Date column index in CSV (0-based)
            time=-1,                    # Time column index (not present, or not used here)
            high=2,                     # High price column index
            low=3,                      # Low price column index
            open=1,                     # Open price column index
            close=4,                    # Close price column index
            volume=5,                   # Volume column index
            openinterest=-1             # Open Interest column index (not present, or not used here)
        )
        cerebro.adddata(data) # Add the configured data feed to the Cerebro engine

    # --- Main Strategy Logic: Executed on Each New Data Bar (e.g., each day) ---
    def next(self):
        super(MyStrategy, self).next() # Call the parent class's next method

        # --- Get Current Market Data ---
        dt = self.datas[0].datetime.datetime(0) # Get the current date/time from the primary data feed
        cl = self.dataclose[0]                 # Get the current day's closing price

        # --- Prepare Input for the Machine Learning Model ---
        inputRec = [] # List to store the final input features for the ML model

        # Temporary list to hold close and SMA values for normalization
        inputRec0 = []
        inputRec0.append(cl) # Add current closing price

        # Collect SMA values, handling NaN (Not a Number) by using current close price
        for a in range(0, self.repeatCount):
            if math.isnan(self.sma[a][0]): # Check if the SMA value is NaN (means not enough data yet)
                inputRec0.append(cl)       # If NaN, use current close price as a fallback
            else:
                inputRec0.append(self.sma[a][0]) # Otherwise, use the calculated SMA value

        # --- Min-Max Normalization for Close and SMA Values ---
        # Scale values to a range between 0 and 1, as the ML model expects normalized inputs
        m1 = min(inputRec0) # Find the minimum value among current close and SMAs
        m2 = max(inputRec0) # Find the maximum value
        for a in inputRec0:
            if m2 - m1 == 0: # Avoid division by zero if all values are the same
                inputRec.append(0)
            else:
                inputRec.append((a - m1) / (m2 - m1)) # Apply min-max normalization formula

        # Collect ROC values, handling NaN by using 0
        for a in range(0, self.repeatCount):
            if math.isnan(self.roc[a][0]): # Check if the ROC value is NaN
                inputRec.append(0)         # If NaN, use 0 as a fallback
            else:
                inputRec.append(self.roc[a][0]) # Otherwise, use the calculated ROC value

        # Initialize mX and dataX (these lines appear redundant or misplaced, as mX is re-initialized below)
        mX = []
        dataX = np.array(mX)

        # --- ML Prediction Section ---
        mX = []
        mX.append(np.array(inputRec)) # Add the prepared input features as a single sample
        # Ensure the input array for the model is at least 2-dimensional (even for a single sample)
        dataX = np.atleast_2d(np.array(mX))

        mY = self.model.predict(dataX) # Get the prediction from the loaded ML model
        mY_squeezed = np.squeeze(mY)   # Remove single-dimensional entries from the shape of the array
                                       # mY will be like [[val1, val2]], squeezed to [val1, val2]

        tLong = mY_squeezed[0]  # The first prediction value (e.g., probability of a 'long' signal)
        tShort = mY_squeezed[1] # The second prediction value (e.g., probability of a 'short' signal)

        # --- Trading Logic: Enter Trades Based on ML Prediction ---
        if not self.position: # Check if there is no open position (we are 'flat')
            # Determine if the ML 'long' prediction confidence is above the configured threshold
            fLong = (tLong > self.config["long_threshold"])
            # Determine if the ML 'short' prediction confidence is above the configured threshold
            fShort = (tShort > self.config["short_threshold"])

            if fLong: # If the 'long' signal is strong enough
                self.order = self.buy(size=self.size) # Place a buy order (go long)
                # Set dynamic profit target and stop-loss prices based on current close
                self.limitPrice = cl + self.profitTarget * cl # Price to sell for profit
                self.stopPrice = cl - self.stopTarget * cl   # Price to sell to cut losses
            elif fShort: # Else if the 'short' signal is strong enough
                self.order = self.sell(size=self.size) # Place a sell order (go short)
                # Set dynamic profit target and stop-loss prices for a short position
                self.limitPrice = cl - self.profitTarget * cl # Price to buy back for profit
                self.stopPrice = cl + self.stopTarget * cl   # Price to buy back to cut losses

        # --- Trading Logic: Manage and Exit Existing Trades ---
        if self.position: # If there is an open position
            if self.position.size > 0: # If currently LONG (positive position size)
                # Check if current price hits profit target or stop-loss
                if cl >= self.limitPrice or cl <= self.stopPrice:
                    self.order = self.sell(size=self.size) # Sell to close the long position
            elif self.position.size < 0: # If currently SHORT (negative position size)
                # Check if current price hits profit target or stop-loss for short
                if cl <= self.limitPrice or cl >= self.stopPrice:
                    self.order = self.buy(size=self.size) # Buy to cover the short position
