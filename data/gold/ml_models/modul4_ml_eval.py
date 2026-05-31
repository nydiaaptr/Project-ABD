# modul4_ml_eval.py
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import time

# Inisialisasi SparkSession
spark = SparkSession.builder \
    .appName("Traffic Volume ML Evaluation") \
    .getOrCreate()

# Path Gold Layer
gold_path = "/home/nydia_manda_putri/spark_project/data/gold/traffic_data"

# Load dataset Gold Layer
df = spark.read.parquet(gold_path)

# Cek schema
df.printSchema()

# Feature columns (sesuaikan dengan Gold Layer)
feature_cols = [col for col in df.columns if col not in ["traffic_volume"]]

# VectorAssembler untuk MLlib
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_ml = assembler.transform(df).select("features", "traffic_volume")

# Split data train/test
train_df, test_df = df_ml.randomSplit([0.8, 0.2], seed=42)

# Evaluator
evaluator_rmse = RegressionEvaluator(labelCol="traffic_volume", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="traffic_volume", predictionCol="prediction", metricName="mae")

print("===== Model Evaluation =====")

# 1. Linear Regression
start_time = time.time()
lr = LinearRegression(featuresCol="features", labelCol="traffic_volume")
lr_model = lr.fit(train_df)
pred_lr = lr_model.transform(test_df)
end_time = time.time()

rmse_lr = evaluator_rmse.evaluate(pred_lr)
mae_lr = evaluator_mae.evaluate(pred_lr)
throughput_lr = test_df.count() / (end_time - start_time)

print(f"Linear Regression - RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}, Throughput: {throughput_lr:.2f} rows/sec, Time: {end_time - start_time:.2f}s")

# 2. Random Forest Regression
start_time = time.time()
rf = RandomForestRegressor(featuresCol="features", labelCol="traffic_volume", numTrees=50)
rf_model = rf.fit(train_df)
pred_rf = rf_model.transform(test_df)
end_time = time.time()

rmse_rf = evaluator_rmse.evaluate(pred_rf)
mae_rf = evaluator_mae.evaluate(pred_rf)
throughput_rf = test_df.count() / (end_time - start_time)

print(f"Random Forest - RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}, Throughput: {throughput_rf:.2f} rows/sec, Time: {end_time - start_time:.2f}s")

# Stop Spark
spark.stop()
