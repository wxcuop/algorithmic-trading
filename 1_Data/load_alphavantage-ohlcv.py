#!/usr/bin/env python3
"""
Alpha Vantage OHLCV Data Loader

This AWS Glue ETL job fetches OHLCV (Open, High, Low, Close, Volume) data 
from Alpha Vantage API and stores it in an Apache Iceberg table for 
algorithmic trading analysis.

Author: Algorithmic Trading Team
Date: 2024
"""

import sys
import logging
from typing import List, Dict, Optional, Any

# AWS Glue imports
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType

# Data processing imports
import pandas as pd
import requests

# =============================================================================
# CONFIGURATION AND PARAMETERS
# =============================================================================

# Get job parameters from AWS Glue
args = getResolvedOptions(sys.argv, ['BUCKET', 'ALPHAVANTAGE_API_KEY', 'GLUE_DATABASE'])

# AWS Configuration
BUCKET_NAME = args['BUCKET']
BUCKET_PREFIX = ""

# Iceberg Table Configuration
ICEBERG_CATALOG_NAME = "glue_catalog"
ICEBERG_DATABASE_NAME = args['GLUE_DATABASE']
ICEBERG_TABLE_NAME = "hist_ohlcv_daily_alphavantage"
WAREHOUSE_PATH = f"s3://{BUCKET_NAME}/{BUCKET_PREFIX}"
FULL_TABLE_NAME = f"{ICEBERG_CATALOG_NAME}.{ICEBERG_DATABASE_NAME}.{ICEBERG_TABLE_NAME}"

# Alpha Vantage API Configuration
API_KEY = args['ALPHAVANTAGE_API_KEY']
SYMBOLS = ['INTC']  # List of stock symbols to fetch OHLCV data for

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# SPARK SESSION CONFIGURATION
# =============================================================================

def create_spark_session() -> SparkSession:
    """
    Create and configure Spark session for Iceberg operations.
    
    Returns:
        SparkSession: Configured Spark session
    """
    return SparkSession.builder \
        .config("spark.sql.warehouse.dir", WAREHOUSE_PATH) \
        .config(f"spark.sql.catalog.{ICEBERG_CATALOG_NAME}", "org.apache.iceberg.spark.SparkCatalog") \
        .config(f"spark.sql.catalog.{ICEBERG_CATALOG_NAME}.warehouse", WAREHOUSE_PATH) \
        .config(f"spark.sql.catalog.{ICEBERG_CATALOG_NAME}.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog") \
        .config(f"spark.sql.catalog.{ICEBERG_CATALOG_NAME}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO") \
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
        .getOrCreate()

# =============================================================================
# DATA SCHEMA DEFINITION
# =============================================================================

# Define the schema for OHLCV data
schema = StructType([
    StructField("dt", DateType(), True),           # Date of the trading day
    StructField("symbol", StringType(), True),     # Stock symbol (e.g., 'INTC')
    StructField("open", DoubleType(), True),       # Opening price
    StructField("high", DoubleType(), True),       # Highest price during the day
    StructField("low", DoubleType(), True),        # Lowest price during the day
    StructField("close", DoubleType(), True),      # Closing price
    StructField("volume", DoubleType(), True)      # Trading volume
])

# =============================================================================
# ALPHA VANTAGE API INTEGRATION
# =============================================================================

class OHLCVFetcher:
    """
    Handles API calls to Alpha Vantage for OHLCV data.
    
    This class manages the interaction with Alpha Vantage's TIME_SERIES_DAILY endpoint,
    including error handling and data validation.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the fetcher with API key.
        
        Args:
            api_key (str): Alpha Vantage API key
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_ohlcv_for_symbol(self, symbol: str) -> Dict[str, Dict[str, str]]:
        """
        Fetches daily OHLCV data for a given symbol.
        
        Args:
            symbol (str): Stock symbol to fetch data for
            
        Returns:
            Dict[str, Dict[str, str]]: Dictionary with dates as keys and OHLCV data as values
        """
        logger.info(f"Fetching OHLCV data for symbol: {symbol}")
        
        # Construct API URL for daily time series
        url = (
            f'{self.base_url}?function=TIME_SERIES_DAILY'
            f'&symbol={symbol}&apikey={self.api_key}'
            f'&outputsize=full'  # Get full historical data
        )
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                logger.info(f"Successfully fetched OHLCV data for {symbol}")
                data = response.json()
                return data.get('Time Series (Daily)', {})
            else:
                logger.error(f"Failed to fetch OHLCV for {symbol}, status code: {response.status_code}")
                return {}
        except requests.RequestException as e:
            logger.error(f"Request failed for {symbol}: {e}")
            return {}

# =============================================================================
# DATA PROCESSING CLASSES
# =============================================================================

class OHLCVRecord:
    """
    Represents a single day's OHLCV data point.
    
    This class processes raw OHLCV data from Alpha Vantage and converts
    string values to appropriate numeric types for analysis.
    """
    
    def __init__(self, date: str, values: Dict[str, str], symbol: str):
        """
        Initialize an OHLCV record from raw data.
        
        Args:
            date (str): Date string in YYYY-MM-DD format
            values (Dict[str, str]): Raw OHLCV values from Alpha Vantage
            symbol (str): Stock symbol this record is for
        """
        self.symbol = symbol
        self.date = date
        
        # Extract OHLCV values from Alpha Vantage response
        # Alpha Vantage returns these as strings, we'll convert to float
        self.open = values.get('1. open')
        self.high = values.get('2. high')
        self.low = values.get('3. low')
        self.close = values.get('4. close')
        self.volume = values.get('5. volume')

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the OHLCV record to a dictionary for DataFrame creation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the OHLCV record
        """
        return {
            'dt': self.date,
            'symbol': self.symbol,
            'open': float(self.open) if self.open else None,
            'high': float(self.high) if self.high else None,
            'low': float(self.low) if self.low else None,
            'close': float(self.close) if self.close else None,
            'volume': float(self.volume) if self.volume else None,
        }

# =============================================================================
# MAIN EXECUTION LOGIC
# =============================================================================

def main():
    """Main execution function for the ETL job."""
    
    logger.info("Starting Alpha Vantage OHLCV ETL Job")
    
    # Initialize Spark session
    logger.info("Initializing Spark session...")
    spark = create_spark_session()
    logger.info("Spark session created successfully")
    
    # Initialize data fetcher
    fetcher = OHLCVFetcher(API_KEY)
    ohlcv_records = []
    
    # Process each symbol
    logger.info(f"Processing {len(SYMBOLS)} symbol(s): {SYMBOLS}")
    
    for symbol in SYMBOLS:
        logger.info(f"Processing symbol: {symbol}")
        
        # Fetch OHLCV data for the symbol
        ohlcv_data = fetcher.fetch_ohlcv_for_symbol(symbol)
        
        if not ohlcv_data:
            logger.warning(f"No OHLCV data found for symbol: {symbol}")
            continue
        
        # Process each daily record
        for date, values in ohlcv_data.items():
            try:
                ohlcv_record = OHLCVRecord(date=date, values=values, symbol=symbol)
                ohlcv_records.append(ohlcv_record.to_dict())
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to process OHLCV record for {symbol} on {date}: {e}")
                continue
    
    logger.info(f"Fetched and processed OHLCV records for {len(SYMBOLS)} symbol(s). "
                f"Total records: {len(ohlcv_records)}")
    
    if not ohlcv_records:
        logger.warning("No OHLCV records found. Exiting without writing to table.")
        return
    
    # Create Pandas DataFrame
    logger.info("Creating Pandas DataFrame from OHLCV records...")
    pandas_df = pd.DataFrame(ohlcv_records)
    
    # Convert date column to proper date format
    pandas_df['dt'] = pd.to_datetime(pandas_df['dt']).dt.date
    
    # Convert Pandas DataFrame to Spark DataFrame
    logger.info("Converting Pandas DataFrame to Spark DataFrame...")
    spark_df = spark.createDataFrame(pandas_df, schema=schema)
    
    # Write data to Iceberg table
    try:
        logger.info(f"Attempting to append OHLCV data to Iceberg table: {FULL_TABLE_NAME}")
        spark_df.writeTo(FULL_TABLE_NAME).append()
        logger.info("OHLCV data appended successfully to existing table.")
    except Exception as append_err:
        logger.warning(f"Append failed, attempting to create table: {append_err}")
        try:
            logger.info(f"Attempting to create Iceberg table: {FULL_TABLE_NAME}")
            (
                spark_df.writeTo(FULL_TABLE_NAME)
                .using("iceberg")
                .tableProperty("format-version", "2")  # Use Iceberg format version 2
                .partitionedBy(year(col("dt")), month(col("dt")), day(col("dt")))  # Partition by date
                .create()
            )
            logger.info(f"Table {FULL_TABLE_NAME} created successfully with date partitioning.")
        except Exception as create_err:
            logger.error(f"Failed to create Iceberg table: {create_err}")
            raise
    
    logger.info("Alpha Vantage OHLCV ETL Job completed successfully!")

if __name__ == "__main__":
    main()
