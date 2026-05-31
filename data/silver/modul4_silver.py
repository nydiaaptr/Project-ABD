from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, avg

spark = SparkSession.builder.appName("SilverLayerTraffic").getOrCreate()

df = spark.read.parquet("~/spark_project/data/bronze/traffic_data")

# Feature engineering sederhana
df_silver = df.withColumn("hour", hour("date_time"))
df_silver = df_silver.groupBy("hour").agg(avg("traffic_volume").alias("avg_traffic"))

df_silver.write.mode("overwrite").parquet("~/spark_project/data/silver/traffic_data")
print("Silver Layer selesai!")
