from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, avg

spark = SparkSession.builder.appName("SilverLayer").getOrCreate()

# Load Bronze data
df_bronze = spark.read.parquet("bronze/traffic_data")

# Tambahkan kolom hour
df_silver = df_bronze.withColumn("hour", hour("date_time"))

# Ambil kolom penting
df_silver = df_silver.select("hour", "traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all", "holiday")

# Agregasi per jam
df_silver = df_silver.groupBy("hour").agg(
    avg("traffic_volume").alias("avg_traffic"),
    avg("temp").alias("avg_temp"),
    avg("rain_1h").alias("avg_rain"),
    avg("snow_1h").alias("avg_snow"),
    avg("clouds_all").alias("avg_clouds")
)

# Simpan ke Silver
df_silver.write.mode("overwrite").parquet("silver/traffic_data")

print("Silver Layer selesai!")
spark.stop()
