import boto3
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import *
from awsglue.context import GlueContext
from awsglue.job import Job
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd

args = getResolvedOptions(sys.argv,['BUCKET','ALPHAVANTAGE_API_KEY'])
BUCKET_NAME = args['BUCKET']
BUCKET_PREFIX = ""
ICEBERG_CATALOG_NAME = "glue_catalog"
ICEBERG_DATABASE_NAME = "algo_data"
ICEBERG_TABLE_NAME = "hist_news_daily_alphavantage"
WAREHOUSE_PATH = f"s3://{BUCKET_NAME}/{BUCKET_PREFIX}"
FULL_TABLE_NAME = f"{ICEBERG_CATALOG_NAME}.{ICEBERG_DATABASE_NAME}.{ICEBERG_TABLE_NAME}"
START_DATE = datetime(2021, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2024, 12, 31, tzinfo=timezone.utc)
INTERVAL_DAYS = 180 # Fetch data in 180-day intervals
API_KEY = args['ALPHAVANTAGE_API_KEY']
SYMBOLS = ['INTC', 'AMD', 'NVDA']

from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .config("spark.sql.warehouse.dir", WAREHOUSE_PATH) \
    .config(f"spark.sql.catalog.{ICEBERG_CATALOG_NAME}", "org.apache.iceberg.spark.SparkCatalog") \
    .config(f"spark.sql.catalog.{ICEBERG_CATALOG_NAME}.warehouse", WAREHOUSE_PATH) \
    .config(f"spark.sql.catalog.{ICEBERG_CATALOG_NAME}.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog") \
    .config(f"spark.sql.catalog.{ICEBERG_CATALOG_NAME}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .getOrCreate()

# sc = SparkContext.getOrCreate()

# glueContext = GlueContext(sc)
# spark = glueContext.spark_session
# job = Job(spark)
# 4. Define schema for Spark DataFrame (Good practice for Iceberg)
# This helps ensure correct data types in Iceberg, especially for `date`.
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType
schema = StructType([
    StructField("symbol", StringType(), True),
    StructField("time_published_datetime", DateType(), True),
    StructField("sentiment_score", DoubleType(), True)
])

# --- Configuration ---
API_KEY = 'Z3TBUBW7GS7WSE2W' # Replace with your actual Alpha Vantage API Key
SYMBOLS = ['INTC', 'AMD', 'NVDA']
START_DATE = datetime(2021, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2024, 12, 31, tzinfo=timezone.utc)
INTERVAL_DAYS = 180 # Fetch data in 180-day intervals

# --- Data Fetching and Processing Classes (from your original code) ---
class NewsSentimentFetcher:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_news_for_symbol_in_interval(self, symbol, time_from, time_to):
        """Fetches news for a given symbol within a specified time range."""
        time_from_str = time_from.strftime('%Y%m%dT%H%M')
        time_to_str = time_to.strftime('%Y%m%dT%H%M')

        url = (
            f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT'
            f'&tickers={symbol}&apikey={self.api_key}'
            f'&time_from={time_from_str}&time_to={time_to_str}&limit=1000'
        )
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get('feed', [])
        else:
            print(f"Failed to fetch news for {symbol} from {time_from_str} to {time_to_str}. Status Code: {response.status_code}")
            return []

class NewsRecord:
    def __init__(self, symbol, article):
        self.symbol = symbol
        self.time_published = article.get('time_published') # Original string
        self.sentiment_score = self.extract_sentiment_score(article, symbol)

        # Convert time_published string to datetime object
        self.published_datetime = self._parse_time_published_to_datetime(self.time_published)
        
        # Extract date string (YYYY-MM-DD) from the datetime object
        self.date = self._extract_date_str(self.published_datetime)

    @staticmethod
    def _parse_time_published_to_datetime(time_published_str):
        """Parses a YYYYMMDDTHHMMSS string into a datetime object."""
        if time_published_str:
            try:
                # Alpha Vantage time_published format is YYYYMMDDTHHMMSS
                return datetime.strptime(time_published_str, '%Y%m%dT%H%M%S')
            except ValueError:
                # Handle cases where seconds might be missing or format is slightly different
                try:
                    return datetime.strptime(time_published_str, '%Y%m%dT%H%M')
                except Exception:
                    return None
        return None

    @staticmethod
    def _extract_date_str(dt_obj):
        """Extracts date in YYYY-MM-DD format from a datetime object."""
        if dt_obj:
            return dt_obj.strftime('%Y-%m-%d')
        return None

    @staticmethod
    def extract_sentiment_score(article, symbol):
        for ts in article.get('ticker_sentiment', []):
            if ts.get('ticker') == symbol:
                try:
                    return float(ts.get('ticker_sentiment_score'))
                except (TypeError, ValueError):
                    return None
        return None

    def to_dict(self):
        return {
            'symbol': self.symbol,
            'time_published_datetime': self.published_datetime, # Datetime object
            'sentiment_score': self.sentiment_score,
        }

fetcher = NewsSentimentFetcher(API_KEY)
news_records = []

# Iterate through the time intervals
current_start_date = START_DATE
while current_start_date <= END_DATE:
    current_end_date = current_start_date + timedelta(days=INTERVAL_DAYS)
    if current_end_date > END_DATE:
        current_end_date = END_DATE

    print(f"Fetching news from {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")

    for symbol in SYMBOLS:
        news = fetcher.fetch_news_for_symbol_in_interval(symbol, current_start_date, current_end_date)
        for article in news:
            record = NewsRecord(symbol=symbol, article=article)
            if record.sentiment_score is not None:
                news_records.append(record.to_dict())
    
    current_start_date = current_end_date + timedelta(days=1) # Move to the next day to avoid overlaps

# 4. Create Pandas DataFrame
pandas_df = pd.DataFrame(news_records)

# 5. Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(pandas_df, schema=schema)

try:
    # Try to append first
    spark_df.writeTo(FULL_TABLE_NAME).append()
except Exception as append_err:
    print(f"Append failed, attempting to create table: {append_err}")
    try:
        (
            spark_df.writeTo(FULL_TABLE_NAME)
            .using("iceberg")
            .partitionedBy("time_published_datetime")  # Partitioning
            .tableProperty("format-version", "2")  # Optional Iceberg version
            .create()
        )
        print(f"Table {FULL_TABLE_NAME} created successfully.")
    except Exception as create_err:
        print(f"Failed to create Iceberg table: {create_err}")