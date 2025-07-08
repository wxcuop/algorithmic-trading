import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
import requests
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType


args = getResolvedOptions(sys.argv,['BUCKET','ALPHAVANTAGE_API_KEY', "GLUE_DATABASE"])
BUCKET_NAME = args['BUCKET']
BUCKET_PREFIX = ""
ICEBERG_CATALOG_NAME = "glue_catalog"
ICEBERG_DATABASE_NAME = args['GLUE_DATABASE']
ICEBERG_TABLE_NAME = "hist_ohlcv_daily_alphavantage"
WAREHOUSE_PATH = f"s3://{BUCKET_NAME}/{BUCKET_PREFIX}"
FULL_TABLE_NAME = f"{ICEBERG_CATALOG_NAME}.{ICEBERG_DATABASE_NAME}.{ICEBERG_TABLE_NAME}"
API_KEY = args['ALPHAVANTAGE_API_KEY']
SYMBOLS = ['INTC']

spark = SparkSession.builder \
    .config("spark.sql.warehouse.dir", WAREHOUSE_PATH) \
    .config(f"spark.sql.catalog.{ICEBERG_CATALOG_NAME}", "org.apache.iceberg.spark.SparkCatalog") \
    .config(f"spark.sql.catalog.{ICEBERG_CATALOG_NAME}.warehouse", WAREHOUSE_PATH) \
    .config(f"spark.sql.catalog.{ICEBERG_CATALOG_NAME}.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog") \
    .config(f"spark.sql.catalog.{ICEBERG_CATALOG_NAME}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .getOrCreate()

schema = StructType([
    StructField("dt", DateType(), True),   
    StructField("symbol", StringType(), True),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", DoubleType(), True)        
])

# --- Data Fetching and Processing Classes (from your original code) ---
class OHLCVFetcher:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_ohlcv_for_symbol(self, symbol):
        time_from, time_to = self.get_time_range_str()
        url = (
            f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY'
            f'&symbol={symbol}&apikey={self.api_key}'
            f'&outputsize=full'

        )
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get('Time Series (Daily)', [])
        else:
            print(f"Failed to fetch OHLCV for {symbol}")
            return []

class OHLCVRecord:
    def __init__(self, date, values, symbol):
        self.symbol = symbol
        self.open = values.get('1. open') # Original string
        self.high = values.get('2. high') # Original string
        self.low = values.get('3. low') # Original string
        self.close = values.get('4. close') # Original string
        self.volume = values.get('5. volume') # Original string        
        # Extract date string (YYYY-MM-DD) from the datetime object
        self.date = date

    def to_dict(self):
        return {
            'dt': self.date, # YYYY-MM-DD string - Removed as requested
            'symbol': self.symbol,
            'open': float(self.open),
            'high': float(self.high), # Datetime object
            'low': float(self.low), # Datetime object
            'close': float(self.close),
            'volume': float(self.volume),            
        }


fetcher = OHLCVFetcher(API_KEY)
ohlcv_records = []

for symbol in SYMBOLS:
    ohlcv = fetcher.fetch_ohlcv_for_symbol(symbol)

for date, values in ohlcv.items():
    ohlcv_tick = OHLCVRecord(date=date, values = values, symbol = symbol)
    ohlcv_records.append(ohlcv_tick.to_dict())

# 4. Create Pandas DataFrame
pandas_df = pd.DataFrame(ohlcv_records)
pandas_df['dt'] = pd.to_datetime(pandas_df['dt']).dt.date

# 5. Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(pandas_df, schema=schema)

try:
    # Try to append first
    spark_df.writeTo(FULL_TABLE_NAME).append()
except Exception as append_err:
    try:
        (
            spark_df.writeTo(FULL_TABLE_NAME)
            .using("iceberg")
            .tableProperty("format-version", "2")  # Optional Iceberg version
            .partitionedBy("days(dt)")             # <-- Partition by day
            .create()
        )
    except Exception as create_err:
        print(f"Failed to create Iceberg table: {create_err}")
