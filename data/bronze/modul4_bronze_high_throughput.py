from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Inisialisasi SparkSession
spark = SparkSession.builder \
    .appName("BronzeLayerHighThroughput") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Path CSV (pastikan sesuai lokasi di WSL)
csv_path = "/mnt/c/Users/NYDIA\\ MANDA\\ PUTRI/Downloads/Metro_Interstate_Traffic_Volume.csv"

# Load CSV dengan inferSchema dan header
df = spark.read.csv(csv_path, header=True, inferSchema=True)

# Optimisasi: ubah tipe data jika perlu untuk meningkatkan kecepatan
df = df.withColumn("traffic_volume", col("traffic_volume").cast("integer"))

# Simpan ke Bronze Layer dalam format Parquet (lebih cepat daripada CSV)
bronze_path = "~/spark_project/data/bronze/traffic_data"
df.write.mode("overwrite").parquet(bronze_path)

# Print jumlah baris dan beberapa sample
print("Jumlah baris:", df.count())
df.show(5, truncate=False)

# Stop SparkSession
spark.stop()
