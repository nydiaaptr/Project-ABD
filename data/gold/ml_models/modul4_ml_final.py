from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import time

spark = SparkSession.builder.appName("GoldLayerTraffic_ML").getOrCreate()

# Load Silver
df = spark.read.parquet("~/spark_project/data/silver/traffic_data")

# Feature vector
assembler = VectorAssembler(inputCols=["hour", "avg_traffic"], outputCol="features")
df = assembler.transform(df)

# Split train/test
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# Random Forest model
rf = RandomForestRegressor(featuresCol="features", labelCol="avg_traffic", numTrees=50)

start_time = time.time()
rf_model = rf.fit(train_df)
train_time = time.time() - start_time

# Predict
predictions = rf_model.transform(test_df)

# Evaluation
evaluator_rmse = RegressionEvaluator(labelCol="avg_traffic", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="avg_traffic", predictionCol="prediction", metricName="mae")
rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
throughput = test_df.count() / train_time

print("===== Model Evaluation =====")
print(f"Random Forest - RMSE: {rmse:.2f}, MAE: {mae:.2f}, Throughput: {throughput:.2f} rows/sec, Time: {train_time:.2f}s")

# Save predictions
predictions.select("avg_traffic", "prediction").write.mode("overwrite").parquet("~/spark_project/data/gold/predictions/predictions_rf")

spark.stop()
