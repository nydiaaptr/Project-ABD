from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

# Inisialisasi Spark
spark = SparkSession.builder.appName("GoldLayerTraffic").getOrCreate()

# Load data dari Silver Layer
df = spark.read.parquet("/home/nydia_manda_putri/spark_project/data/silver/traffic_data")

# Contoh agregasi: rata-rata traffic per jam
df_gold = df.groupBy("hour").agg(avg("traffic_volume").alias("avg_traffic"))

# Simpan hasil ke Gold Layer
df_gold.write.mode("overwrite").parquet("/home/nydia_manda_putri/spark_project/data/gold/traffic_data")

print("Gold Layer selesai!")
