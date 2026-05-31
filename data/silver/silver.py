from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

spark = SparkSession.builder.appName("SilverLayerTraffic").getOrCreate()

# Load Bronze
df = spark.read.parquet("~/spark_project/data/bronze/traffic_data")

# Feature engineering: ambil rata-rata per jam
df_silver = df.groupBy("hour").agg(avg("traffic_volume").alias("avg_traffic"))

# Simpan ke Silver
df_silver.write.mode("overwrite").parquet("~/spark_project/data/silver/traffic_data")
print("Silver Layer selesai!")
spark.stop()
