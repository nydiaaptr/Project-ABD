from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

spark = SparkSession.builder.appName("GoldLayer").getOrCreate()

# Load Silver data
df_silver = spark.read.parquet("silver/traffic_data")

# Buat features
feature_cols = ["avg_temp", "avg_rain", "avg_snow", "avg_clouds", "hour"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_gold = assembler.transform(df_silver).select("features", "avg_traffic")

# Split train/test
train_df, test_df = df_gold.randomSplit([0.7, 0.3], seed=42)

# Random Forest
rf = RandomForestRegressor(featuresCol="features", labelCol="avg_traffic", numTrees=100, maxDepth=5)
rf_model = rf.fit(train_df)
predictions = rf_model.transform(test_df)

# Evaluasi
evaluator_rmse = RegressionEvaluator(labelCol="avg_traffic", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="avg_traffic", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="avg_traffic", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

pred_pd = predictions.select("avg_traffic", "prediction").toPandas()
pred_pd["mape"] = np.abs(pred_pd["avg_traffic"] - pred_pd["prediction"]) / pred_pd["avg_traffic"]
mape = pred_pd["mape"].mean() * 100

print("===== Model Evaluation =====")
print(f"Random Forest - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")

# LOESS Smoothed Curve
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
