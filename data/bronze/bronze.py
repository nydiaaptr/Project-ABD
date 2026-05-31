from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp

spark = SparkSession.builder.appName("BronzeLayerTraffic").getOrCreate()

# Load CSV
df = spark.read.csv("/mnt/c/Users/NYDIA MANDA PUTRI/Downloads/Metro_Interstate_Traffic_Volume.csv",
                    header=True, inferSchema=True)

# Cleaning
df = df.dropDuplicates()
df = df.drop("weather_description")  # contoh drop kolom yang tidak diperlukan
df = df.withColumn("date_time", to_timestamp("date_time", "yyyy-MM-dd HH:mm:ss"))

# Simpan ke Bronze
df.write.mode("overwrite").parquet("~/spark_project/data/bronze/traffic_data")
print("Bronze Layer selesai!")
spark.stop()
