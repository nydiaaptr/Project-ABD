from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, col, avg
from pyspark.sql.functions import to_timestamp
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# =====================
# Bronze Layer: Load Raw Data
# =====================
spark = SparkSession.builder.appName("TrafficPrediction").getOrCreate()
bronze_path = "/data/Metro_Interstate_Traffic_Volume.csv"
df_bronze = spark.read.csv(bronze_path, header=True, inferSchema=True)

# Ubah date_time menjadi timestamp
df_bronze = df_bronze.withColumn("date_time", to_timestamp("date_time", "yyyy-MM-dd HH:mm:ss"))

# =====================
# Silver Layer: Preprocessing & Feature Engineering
# =====================
# Buat kolom 'hour'
df_silver = df_bronze.withColumn("hour", hour("date_time"))

# Ambil kolom yang penting
df_silver = df_silver.select("hour", "traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all", "holiday")

# Agregasi per jam
df_silver = df_silver.groupBy("hour").agg(
    avg("traffic_volume").alias("avg_traffic"),
    avg("temp").alias("avg_temp"),
    avg("rain_1h").alias("avg_rain"),
    avg("snow_1h").alias("avg_snow"),
    avg("clouds_all").alias("avg_clouds")
)

# =====================
# Gold Layer: ML Modeling
# =====================
# Features
feature_cols = ["avg_temp", "avg_rain", "avg_snow", "avg_clouds", "hour"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_gold = assembler.transform(df_silver).select("features", "avg_traffic")

# Split train/test
train_df, test_df = df_gold.randomSplit([0.7, 0.3], seed=42)

# Model Random Forest
rf = RandomForestRegressor(featuresCol="features", labelCol="avg_traffic", numTrees=100, maxDepth=5)
rf_model = rf.fit(train_df)
predictions = rf_model.transform(test_df)

# Evaluator
evaluator_rmse = RegressionEvaluator(labelCol="avg_traffic", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="avg_traffic", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="avg_traffic", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

# MAPE
pred_pd = predictions.select("avg_traffic", "prediction").toPandas()
pred_pd["mape"] = np.abs(pred_pd["avg_traffic"] - pred_pd["prediction"]) / pred_pd["avg_traffic"]
mape = pred_pd["mape"].mean() * 100

# =====================
# Print Evaluation
# =====================
print("===== Model Evaluation =====")
print(f"Random Forest - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")

# =====================
# Plot LOESS-style fit (smoothed)
# =====================
from statsmodels.nonparametric.smoothers_lowess import lowess

loess_smoothed = lowess(pred_pd["prediction"], pred_pd.index, frac=0.3)
plt.figure(figsize=(10,6))
plt.plot(pred_pd.index, pred_pd["avg_traffic"], "o", label="Actual")
plt.plot(loess_smoothed[:,0], loess_smoothed[:,1], "r-", label="LOESS Smoothed")
plt.xlabel("Observation")
plt.ylabel("Traffic Volume")
plt.title("Random Forest Predictions - LOESS Smoothed")
plt.legend()
plt.show()

spark.stop()
