from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp

spark = SparkSession.builder.appName("BronzeLayer").getOrCreate()

# Path CSV raw data
bronze_path = "/mnt/c/Users/NYDIA MANDA PUTRI/Downloads/Metro_Interstate_Traffic_Volume.csv"

# Load CSV
df_bronze = spark.read.csv(bronze_path, header=True, inferSchema=True)
df_bronze = df_bronze.withColumn("date_time", to_timestamp("date_time", "yyyy-MM-dd HH:mm:ss"))

# Simpan ke Bronze folder
df_bronze.write.mode("overwrite").parquet("bronze/traffic_data")

print("Bronze Layer selesai!")
spark.stop()
