# Algorithmic Trading Proof-of-Concept

This proof-of-concept (POC) demonstrates a machine learning-based quantitative research environment. It shows how to load and store financial data on AWS (from AWS Data Exchange and other external sources), and how to build and backtest algorithmic trading strategies with Amazon SageMaker using technical indicators and advanced machine learning models.

> **Note:**  
> This is a demo, not a production trading algorithm. Training is performed on synthetic data. The objective is to showcase the quant platform setup, not to demonstrate alpha generation.

![Strategy Chart](assets/chart.png)

---

## Supported AWS Region

- **us-east-1** (tested)

---

## Team Roles & Responsibilities

As a solo team lead, you'll wear multiple hats:

- **Data Engineer:** Modify scripts to load external market data into AWS.
- **Data Scientist:** Load data into your ML environment, analyze it, train models, and make predictions.
- **Trader:** Use different trading strategies to maximize profit & loss while managing risk.

---

## Project Goals

By the end of this POC, you will:

- Understand how to load historical price data from external sources (like AWS Data Exchange) into S3.
- Learn to store price data in S3, expose it via Glue Data Catalog and Athena, and backtest trading strategies using Amazon SageMaker.
- Train machine learning models for use in trading strategies.
- Gain a basic understanding of developing and optimizing trend-following and ML-based trading strategies in Python.

---

## Architecture Overview

1. **Environment Setup:**  
   Use CloudFormation to create the environment (clusters, SageMaker, roles, etc.).

2. **Data Generation:**  
   Generate synthetic data with `1_Data/Load_Hist_Data_Daily_Public.ipynb` and store it in S3.

3. **Data Cataloging:**  
   Use AWS Glue to crawl the data and create a table, queryable via Athena.

4. **Strategy Backtesting:**  
   - Non-ML strategies (SMA, Breakout) use the data directly.
   - ML-based strategies (`2_Strategies/Strategy_Forecast.ipynb`) involve an intermediate model training step (`3_Models/Train_Model_Forecast.ipynb`).

5. **Parameter Optimization:**  
   Each strategy has a hyperparameter file for running various parameter combinations during backtesting.

6. **Containerization:**  
   Strategies are containerized and can run locally or on an Elastic Container cluster. Results (files and plots) are generated in both cases.

7. **Logging:**  
   Execution logs are sent to AWS CloudWatch.

![Architecture Diagram](assets/algo-trading-diagram.drawio.png)


---

## Acknowledgements

This work is based on the AWS blog post:  
[Algorithmic Trading with SageMaker and AWS Data Exchange (Feb 2021)](https://aws.amazon.com/blogs/industries/algorithmic-trading-on-aws-with-amazon-sagemaker-and-aws-data-exchange/)  
Thanks to the authors for their ideas and code contributions.

---

## Quick Start: SageMaker Notebooks

### Step 0: Set Up Environment

1. **Create S3 Bucket:**  
   Create a unique S3 bucket starting with `algotrading-` (e.g., `algotrading-YYYY-MM-DD-XYZ`) for storing external price data.

2. **Deploy Infrastructure:**  
   Deploy the [CloudFormation template](https://github.com/aws-samples/algorithmic-trading/raw/master/0_Setup/algo-reference.yaml) for SageMaker Notebook, Athena, and Glue Tables.  
   - Go to [CloudFormation](https://console.aws.amazon.com/cloudformation/home?#/stacks/new?stackName=algotrading) and upload the template.
   - Specify your S3 bucket name.
   - Ensure the stack name is `algotrading` and acknowledge IAM changes.

---

### Step 1: Load Historical Price Data

- Generate sample EOD price data from a public source.
- Run all cells in `1_Data/Load_Hist_Data_Daily_Public.ipynb`.

---

### Step 2: Backtest a Trend-Following Strategy

Backtest a trend-following strategy on daily price data with Amazon SageMaker. Ensure daily price data is loaded.

**Available Strategies:**

- **Simple Moving Average (SMA) Strategy:**  `2_Strategies/Strategy SMA.ipynb`  
  - Buys when a fast SMA crosses above a slow SMA (uptrend).
  - Sells when a fast SMA crosses below a slow SMA (downtrend).
  - Reverses position on opposing signals.

- **Daily Breakout Strategy:**  `2_Strategies/Strategy_Breakout.ipynb`  
  - Buys when the price exceeds the highest high of the look-back period.
  - Sells when the price falls below the lowest low of the look-back period.
  - Configurable to go long, short, or both.

> **Tip:**  
> Select the appropriate Jupyter Notebook in `2_Strategies` and follow the instructions for strategy optimization.

---

### Step 3: Backtest a Machine Learning-Based Strategy

Backtest an ML-based strategy with Amazon SageMaker on daily or intraday price data.

- **Training:** `3_Models/Train_Model_Forecast.ipynb`  
    - Multilayer Perceptron (MLP) with 3 layers (input, hidden, output).
    - Binary classification:  
      - **Input:** Close, SMA(2–16), ROC(2–16)  
      - **Output:** Will a long/short trade hit a 2% profit target without hitting a 1.5% stop loss in the next 5 days?
    - Data is normalized and labeled for ML.

- **Backtesting:** `2_Strategies/Strategy_Forecast.ipynb`  
    - Loads a pre-trained Keras model.
    - Prepares input features (normalized close, SMAs, ROCs).
    - Predicts long/short opportunities.
    - Executes trades based on predictions and configurable thresholds.
    - Manages risk with profit targets and stop-losses.

---

## Strategy Details

### 1. Simple Moving Average (SMA) Crossover Strategy

This classic trend-following strategy uses two Simple Moving Averages (SMAs) with different periods:

- **Fast SMA:** Shorter period, reacts quickly to price changes.

- **Slow SMA:** Longer period, smooths out price fluctuations.

**Logic:**

- **Go Long (Buy):** When the fast SMA crosses above the slow SMA, signaling an uptrend.

- **Go Short (Sell):** When the fast SMA crosses below the slow SMA, signaling a downtrend.

- **Position Reversal:** If an opposing signal occurs while a position is open, the strategy reverses the position to align with the new trend direction.

**Risk Management:**

- No explicit stop-loss or take-profit; the strategy relies on signal reversals for exits.

**Configuration:**

- `fast_period`: Period for the fast SMA.

- `slow_period`: Period for the slow SMA.

- `size`: Trade size per signal.

---

### 2. Daily Breakout Strategy

A momentum-based strategy that seeks to capture strong price movements by identifying breakouts from recent highs or lows.

**Logic:**

  - **Go Long (Buy):** When the current price exceeds the highest high over a configurable look-back period.

  - **Go Short (Sell):** When the current price falls below the lowest low over the look-back period.

  - **Exit Long:** If the price drops below the previous period's high.

  - **Exit Short:** If the price rises above the previous period's low.

- The strategy can be configured to only go long, only go short, or both.

**Risk Management:**

  - Exits are triggered by price reversals relative to the breakout levels.

**Configuration:**

  - `period`: Look-back period for high/low calculation.

  - `go_long`: Enable/disable long trades.

  - `go_short`: Enable/disable short trades.

  - `size`: Trade size per signal.

---

### 3. Machine Learning Long/Short Prediction Strategy

An advanced strategy that leverages a pre-trained neural network (MLP) to predict the probability of profitable long or short trades, using technical indicators as features.

**Logic:**

- **Feature Engineering:**

    - Inputs: Current close price, multiple SMAs (2–16), and Rates of Change (ROCs) over various periods.

    - SMAs are normalized; ROCs are used as-is.

- **ML Model Prediction:**
    - The model outputs two probabilities: one for a long opportunity, one for a short opportunity.

- **Trade Signals:**

    - **Go Long (Buy):** If the long probability exceeds a configurable threshold.

    - **Go Short (Sell):** If the short probability exceeds a configurable threshold.

- **Position Management:**

    - **Long Position:** Exit if price reaches a profit target (e.g., +2%) or a stop-loss (e.g., -1.5%).

    - **Short Position:** Exit if price reaches a profit target (e.g., -2%) or a stop-loss (e.g., +1.5%).

**Risk Management:**

  - Built-in via configurable profit target and stop-loss percentages.

**Configuration:**

  - `long_threshold`: Probability threshold for entering long trades.

  - `short_threshold`: Probability threshold for entering short trades.

  - `profit_target_pct`: Profit target as a percentage. 

  - `stop_target_pct`: Stop-loss as a percentage.

  - `size`: Trade size per signal.

---
