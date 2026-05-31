from pyspark.sql import SparkSession

# Inisialisasi Spark
spark = SparkSession.builder.appName("ViewPredictions").getOrCreate()

# Load prediksi Linear Regression
df_lr = spark.read.parquet("/home/nydia_manda_putri/spark_project/data/gold/predictions_lr")
print("=== Linear Regression Predictions ===")
df_lr.show(10)  # tampilkan 10 baris pertama

# Load prediksi Random Forest
df_rf = spark.read.parquet("/home/nydia_manda_putri/spark_project/data/gold/predictions_rf")
print("=== Random Forest Predictions ===")
df_rf.show(10)  # tampilkan 10 baris pertama

spark.stop()
