#!/usr/bin/env python3
"""
Alpha Vantage News Sentiment Data Loader

This AWS Glue ETL job fetches news sentiment data from Alpha Vantage API
and stores it in an Apache Iceberg table for algorithmic trading analysis.

Author: Algorithmic Trading Team
Date: 2024
"""

import sys
import logging
from datetime import datetime, timedelta, timezone
from time import sleep
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
args = getResolvedOptions(sys.argv, ['BUCKET', 'ALPHAVANTAGE_API_KEY', 'GLUE_DATABASE', 'EXECUTION_ROLE'])

# AWS Configuration
BUCKET_NAME = args['BUCKET']
BUCKET_PREFIX = ""
EXECUTION_ROLE = args['EXECUTION_ROLE']

# Iceberg Table Configuration
ICEBERG_CATALOG_NAME = "glue_catalog"
ICEBERG_DATABASE_NAME = args['GLUE_DATABASE']
ICEBERG_TABLE_NAME = "hist_news_daily_alphavantage"
WAREHOUSE_PATH = f"s3://{BUCKET_NAME}/{BUCKET_PREFIX}"
FULL_TABLE_NAME = f"{ICEBERG_CATALOG_NAME}.{ICEBERG_DATABASE_NAME}.{ICEBERG_TABLE_NAME}"

# Data Range Configuration
START_DATE = datetime(2022, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 6, 30, tzinfo=timezone.utc)
INTERVAL_DAYS = 90  # Fetch data in 90-day intervals to respect API limits

# Alpha Vantage API Configuration
API_KEY = args['ALPHAVANTAGE_API_KEY']
SYMBOLS = ['INTC']  # List of stock symbols to fetch news for

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

# Define the schema for news sentiment data
schema = StructType([
    StructField("symbol", StringType(), True),                    # Stock symbol (e.g., 'INTC')
    StructField("time_published_datetime", DateType(), True),     # Date when news was published
    StructField("sentiment_score", DoubleType(), True)            # Sentiment score (-1 to 1)
])

# =============================================================================
# ALPHA VANTAGE API INTEGRATION
# =============================================================================

class NewsSentimentFetcher:
    """
    Handles API calls to Alpha Vantage for news sentiment data.
    
    This class manages the interaction with Alpha Vantage's NEWS_SENTIMENT endpoint,
    including rate limiting and error handling.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the fetcher with API key.
        
        Args:
            api_key (str): Alpha Vantage API key
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_news_for_symbol_in_interval(self, symbol: str, time_from: datetime, time_to: datetime) -> List[Dict[str, Any]]:
        """
        Fetches news for a given symbol within a specified time range.
        
        Args:
            symbol (str): Stock symbol to fetch news for
            time_from (datetime): Start of time range
            time_to (datetime): End of time range
            
        Returns:
            List[Dict[str, Any]]: List of news articles with sentiment data
        """
        # Format dates for Alpha Vantage API (YYYYMMDDTHHMM format)
        time_from_str = time_from.strftime('%Y%m%dT%H%M')
        time_to_str = time_to.strftime('%Y%m%dT%H%M')

        # Construct API URL
        url = (
            f'{self.base_url}?function=NEWS_SENTIMENT'
            f'&tickers={symbol}&apikey={self.api_key}'
            f'&time_from={time_from_str}&time_to={time_to_str}&limit=1000'
        )
        
        # Rate limiting: Alpha Vantage allows 5 calls per minute for free tier
        sleep(1)  # Wait 1 second between calls
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                logger.info(f"Successfully fetched news for {symbol} from {time_from_str} to {time_to_str}")
                data = response.json()
                return data.get('feed', [])
            else:
                logger.error(f"Failed to fetch news for {symbol} from {time_from_str} to {time_to_str}. "
                           f"Status Code: {response.status_code}")
                return []
        except requests.RequestException as e:
            logger.error(f"Request failed for {symbol}: {e}")
            return []

# =============================================================================
# DATA PROCESSING CLASSES
# =============================================================================

class NewsRecord:
    """
    Represents a single news article with sentiment data.
    
    This class processes raw news data from Alpha Vantage and extracts
    relevant information including sentiment scores for specific symbols.
    """
    
    def __init__(self, symbol: str, article: Dict[str, Any]):
        """
        Initialize a news record from raw article data.
        
        Args:
            symbol (str): Stock symbol this record is for
            article (Dict[str, Any]): Raw article data from Alpha Vantage
        """
        self.symbol = symbol
        self.time_published = article.get('time_published')  # Original string
        self.sentiment_score = self.extract_sentiment_score(article, symbol)

        # Convert time_published string to datetime object
        self.published_datetime = self._parse_time_published_to_datetime(self.time_published)
        
        # Extract date string (YYYY-MM-DD) from the datetime object
        self.date = self._extract_date_str(self.published_datetime)

    @staticmethod
    def _parse_time_published_to_datetime(time_published_str: str) -> Optional[datetime]:
        """
        Parses a YYYYMMDDTHHMMSS string into a datetime object.
        
        Args:
            time_published_str (str): Time string from Alpha Vantage
            
        Returns:
            Optional[datetime]: Parsed datetime object or None if parsing fails
        """
        if time_published_str:
            try:
                # Alpha Vantage time_published format is YYYYMMDDTHHMMSS
                return datetime.strptime(time_published_str, '%Y%m%dT%H%M%S')
            except ValueError:
                # Handle cases where seconds might be missing or format is slightly different
                try:
                    return datetime.strptime(time_published_str, '%Y%m%dT%H%M')
                except Exception:
                    logger.warning(f"Could not parse time_published: {time_published_str}")
                    return None
        return None

    @staticmethod
    def _extract_date_str(dt_obj: Optional[datetime]) -> Optional[str]:
        """
        Extracts date in YYYY-MM-DD format from a datetime object.
        
        Args:
            dt_obj (Optional[datetime]): Datetime object
            
        Returns:
            Optional[str]: Date string in YYYY-MM-DD format
        """
        if dt_obj:
            return dt_obj.strftime('%Y-%m-%d')
        return None

    @staticmethod
    def extract_sentiment_score(article: Dict[str, Any], symbol: str) -> Optional[float]:
        """
        Extracts sentiment score for a specific symbol from article data.
        
        Args:
            article (Dict[str, Any]): Raw article data
            symbol (str): Stock symbol to extract sentiment for
            
        Returns:
            Optional[float]: Sentiment score (-1 to 1) or None if not found
        """
        for ts in article.get('ticker_sentiment', []):
            if ts.get('ticker') == symbol:
                try:
                    return float(ts.get('ticker_sentiment_score'))
                except (TypeError, ValueError):
                    logger.warning(f"Invalid sentiment score for {symbol}: {ts.get('ticker_sentiment_score')}")
                    return None
        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the news record to a dictionary for DataFrame creation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the news record
        """
        return {
            'symbol': self.symbol,
            'time_published_datetime': self.published_datetime,  # Datetime object
            'sentiment_score': self.sentiment_score,
        }

# =============================================================================
# MAIN EXECUTION LOGIC
# =============================================================================

def main():
    """Main execution function for the ETL job."""
    
    logger.info("Starting Alpha Vantage News Sentiment ETL Job")
    
    # Initialize Spark session
    logger.info("Initializing Spark session...")
    spark = create_spark_session()
    logger.info("Spark session created successfully")
    
    # Initialize data fetcher
    fetcher = NewsSentimentFetcher(API_KEY)
    news_records = []
    
    # Iterate through the time intervals
    logger.info(f"Fetching news data from {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    current_start_date = START_DATE
    
    while current_start_date <= END_DATE:
        current_end_date = current_start_date + timedelta(days=INTERVAL_DAYS)
        if current_end_date > END_DATE:
            current_end_date = END_DATE

        logger.info(f"Processing interval: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")

        # Fetch news for each symbol in the current time interval
        for symbol in SYMBOLS:
            logger.info(f"Fetching news for symbol: {symbol}")
            news = fetcher.fetch_news_for_symbol_in_interval(symbol, current_start_date, current_end_date)
            
            # Process each article
            for article in news:
                record = NewsRecord(symbol=symbol, article=article)
                if record.sentiment_score is not None:
                    news_records.append(record.to_dict())
        
        # Move to the next interval (add 1 day to avoid overlaps)
        current_start_date = current_end_date + timedelta(days=1)
    
    logger.info(f"Total news records collected: {len(news_records)}")
    
    # Create Pandas DataFrame
    logger.info("Creating Pandas DataFrame from news records...")
    pandas_df = pd.DataFrame(news_records)
    
    if pandas_df.empty:
        logger.warning("No news records found. Exiting without writing to table.")
        return
    
    # Convert Pandas DataFrame to Spark DataFrame
    logger.info("Converting Pandas DataFrame to Spark DataFrame...")
    spark_df = spark.createDataFrame(pandas_df, schema=schema)
    
    # Write data to Iceberg table
    try:
        logger.info(f"Attempting to append news data to Iceberg table: {FULL_TABLE_NAME}")
        spark_df.writeTo(FULL_TABLE_NAME).append()
        logger.info("News data appended successfully to existing table.")
    except Exception as append_err:
        logger.warning(f"Append failed, attempting to create table: {append_err}")
        try:
            logger.info(f"Attempting to create Iceberg table: {FULL_TABLE_NAME}")
            (
                spark_df.writeTo(FULL_TABLE_NAME)
                .using("iceberg")
                .tableProperty("format-version", "2")  # Use Iceberg format version 2
                .create()
            )
            logger.info(f"Table {FULL_TABLE_NAME} created successfully.")
        except Exception as create_err:
            logger.error(f"Failed to create Iceberg table: {create_err}")
            raise
    
    logger.info("Alpha Vantage News Sentiment ETL Job completed successfully!")

if __name__ == "__main__":
    main()