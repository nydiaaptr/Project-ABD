from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# 1️⃣ Inisialisasi Spark
spark = SparkSession.builder.appName("TrafficMLFullPipeline").getOrCreate()

# 2️⃣ Load Gold Layer data (pastikan dataset sudah berada di Gold)
data = spark.read.parquet("/home/nydia_manda_putri/spark_project/data/gold/traffic_data")

# 3️⃣ Cek kolom yang ada
data.printSchema()
data.show(5)

# 4️⃣ Tentukan kolom fitur yang ada di dataset
# Sesuaikan dengan kolom yang ada; jika tidak ada 'temp', 'clouds_all', dsb, gunakan kolom yang tersedia
feature_cols = ["hour", "avg_traffic"]  # contoh: hanya pakai kolom hasil agregasi Silver Layer
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# 5️⃣ Split data menjadi train/test
train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)

# 6️⃣ Train model Linear Regression
lr = LinearRegression(featuresCol="features", labelCol="traffic_volume")
lr_model = lr.fit(train_df)
lr_model.save("/home/nydia_manda_putri/spark_project/data/gold/ml_models/lr_model")

# 7️⃣ Train model Random Forest Regression
rf = RandomForestRegressor(featuresCol="features", labelCol="traffic_volume")
rf_model = rf.fit(train_df)
rf_model.save("/home/nydia_manda_putri/spark_project/data/gold/ml_models/rf_model")

# 8️⃣ Prediksi pada test set
predictions_lr = lr_model.transform(test_df)
predictions_rf = rf_model.transform(test_df)

# 9️⃣ Evaluasi model
evaluator_rmse = RegressionEvaluator(labelCol="traffic_volume", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="traffic_volume", predictionCol="prediction", metricName="mae")

rmse_lr = evaluator_rmse.evaluate(predictions_lr)
mae_lr = evaluator_mae.evaluate(predictions_lr)
rmse_rf = evaluator_rmse.evaluate(predictions_rf)
mae_rf = evaluator_mae.evaluate(predictions_rf)

print(f"Linear Regression - RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}")
print(f"Random Forest - RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}")

# 10️⃣ Stop SparkSession
spark.stop()
