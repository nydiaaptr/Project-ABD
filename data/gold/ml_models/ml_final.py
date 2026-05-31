from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, abs as sql_abs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

# Inisialisasi Spark
spark = SparkSession.builder.appName("GoldLayerTraffic_RF").getOrCreate()

# Load Silver layer
df = spark.read.parquet("~/spark_project/data/silver/traffic_data")

# Feature vector
feature_cols = [c for c in df.columns if c != "avg_traffic"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Split data train/test
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Inisialisasi Random Forest
rf = RandomForestRegressor(featuresCol="features", labelCol="avg_traffic",
                           numTrees=50, maxDepth=7, seed=42)

# Fit model
start = time.time()
rf_model = rf.fit(train_df)
predictions = rf_model.transform(test_df)
end = time.time()

# Evaluasi model
evaluator_rmse = RegressionEvaluator(labelCol="avg_traffic", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="avg_traffic", predictionCol="prediction", metricName="mae")
evaluator_r2  = RegressionEvaluator(labelCol="avg_traffic", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
mae  = evaluator_mae.evaluate(predictions)
r2   = evaluator_r2.evaluate(predictions)

# Hitung MAPE
pred_pd = predictions.select("avg_traffic", "prediction").toPandas()
pred_pd["APE"] = np.abs(pred_pd["avg_traffic"] - pred_pd["prediction"]) / pred_pd["avg_traffic"]
mape = pred_pd["APE"].mean() * 100

# Throughput & Latency
throughput = test_df.count() / (end - start)
latency = end - start

print("===== Model Evaluation =====")
print(f"Random Forest - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%, R²: {r2:.2f}")
print(f"Throughput: {throughput:.2f} rows/sec, Time: {latency:.2f}s")

# Learning curve (MSE rata-rata per batch)
pred_pd["MSE"] = (pred_pd["avg_traffic"] - pred_pd["prediction"])**2
plt.figure(figsize=(8,5))
plt.plot(np.arange(len(pred_pd)), pred_pd["MSE"].rolling(10).mean(), color='blue')
plt.title("Loess/Smoothed MSE Curve")
plt.xlabel("Observation")
plt.ylabel("MSE (rolling mean)")
plt.grid(True)
plt.show()

# Simpan prediksi ke Gold/predictions
predictions.select("avg_traffic", "prediction").write.mode("overwrite").parquet(
    "~/spark_project/data/gold/predictions/pred_rf"
)

spark.stop()
