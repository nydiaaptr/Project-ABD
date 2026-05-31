from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# 1️⃣ Inisialisasi Spark
spark = SparkSession.builder.appName("TrafficMLFixed").getOrCreate()

# 2️⃣ Load Gold Layer
# Pastikan path ini sesuai dengan lokasi Gold Layer kamu
data = spark.read.parquet("/home/nydia_manda_putri/spark_project/data/gold/traffic_data")

# 3️⃣ Cek kolom yang ada
data.printSchema()
data.show(5)

# 4️⃣ Sesuaikan kolom fitur dan label sesuai Gold Layer
# Contoh: Gold Layer hanya punya kolom hour dan avg_traffic
feature_cols = ["hour"]  # sesuaikan dengan kolom numerik yang ada
label_col = "avg_traffic"  # gunakan kolom yang ada sebagai label

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# 5️⃣ Split data train/test
train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)

# 6️⃣ Linear Regression
lr = LinearRegression(featuresCol="features", labelCol=label_col)
lr_model = lr.fit(train_df)
lr_model.save("/home/nydia_manda_putri/spark_project/data/gold/ml_models/lr_model")

# 7️⃣ Random Forest Regression
rf = RandomForestRegressor(featuresCol="features", labelCol=label_col)
rf_model = rf.fit(train_df)
rf_model.save("/home/nydia_manda_putri/spark_project/data/gold/ml_models/rf_model")

# 8️⃣ Prediksi
predictions_lr = lr_model.transform(test_df)
predictions_rf = rf_model.transform(test_df)

# 9️⃣ Evaluasi
evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae")

rmse_lr = evaluator_rmse.evaluate(predictions_lr)
mae_lr = evaluator_mae.evaluate(predictions_lr)
rmse_rf = evaluator_rmse.evaluate(predictions_rf)
mae_rf = evaluator_mae.evaluate(predictions_rf)

print(f"Linear Regression - RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}")
print(f"Random Forest - RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}")

# 1️⃣0️⃣ Stop Spark
spark.stop()
