from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import time

spark = SparkSession.builder.appName("GoldLayerTrafficML").getOrCreate()

# Load Silver
df = spark.read.parquet("~/spark_project/data/silver/traffic_data")

# Fitur & Label
assembler = VectorAssembler(inputCols=["hour", "avg_traffic"], outputCol="features")
data = assembler.transform(df).select("features", "avg_traffic")

# Split train-test
train_df, test_df = data.randomSplit([0.7, 0.3], seed=42)

# Linear Regression
lr = LinearRegression(labelCol="avg_traffic", featuresCol="features")
start_time = time.time()
lr_model = lr.fit(train_df)
lr_predictions = lr_model.transform(test_df)
lr_time = time.time() - start_time

# Random Forest
rf = RandomForestRegressor(labelCol="avg_traffic", featuresCol="features", numTrees=50)
start_time = time.time()
rf_model = rf.fit(train_df)
rf_predictions = rf_model.transform(test_df)
rf_time = time.time() - start_time

# Evaluasi
evaluator = RegressionEvaluator(labelCol="avg_traffic", predictionCol="prediction")
lr_rmse = evaluator.evaluate(lr_predictions, {evaluator.metricName: "rmse"})
lr_mae = evaluator.evaluate(lr_predictions, {evaluator.metricName: "mae"})
rf_rmse = evaluator.evaluate(rf_predictions, {evaluator.metricName: "rmse"})
rf_mae = evaluator.evaluate(rf_predictions, {evaluator.metricName: "mae"})

# Hitung throughput
lr_throughput = test_df.count() / lr_time
rf_throughput = test_df.count() / rf_time

print("===== Model Evaluation =====")
print(f"Linear Regression - RMSE: {lr_rmse:.2f}, MAE: {lr_mae:.2f}, Throughput: {lr_throughput:.2f} rows/sec, Time: {lr_time:.2f}s")
print(f"Random Forest - RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}, Throughput: {rf_throughput:.2f} rows/sec, Time: {rf_time:.2f}s")
