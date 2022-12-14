# Databricks notebook source
import numpy as np
import pandas as pd
import random
import uuid
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql.functions import *
from databricks_poc_template import module

# COMMAND ----------

# Creation of a new data batch

# Initialize the dataframe
iris = load_iris()
iris_generated_all = pd.DataFrame(columns=iris.feature_names)

# Generate 50 sample randomly out of each target class
for target_class in [0, 1, 2]:
    iris_generated = module.iris_data_generator(
        target_class=str(target_class), n_samples=50
    )  # module.iris_data_generator...
    iris_generated_all = pd.concat(
        [iris_generated_all, iris_generated], axis=0, ignore_index=True
    )

data_batch = spark.createDataFrame(iris_generated_all)
data_batch = data_batch.withColumnRenamed("sepal length (cm)", "sepal_length")
data_batch = data_batch.withColumnRenamed(
    "sepal width (cm)",
    "sepal_width",
)
data_batch = data_batch.withColumnRenamed("petal length (cm)", "petal_length")
data_batch = data_batch.withColumnRenamed(
    "petal width (cm)",
    "petal_width",
)

# data_batch = data_batch.withColumn('Id', monotonically_increasing_id())
# data_batch = data_batch.withColumn('month',lit('202203'))
data_batch = data_batch.withColumn("date", current_date())
data_batch = data_batch.withColumn("hour", hour(current_timestamp()))
# data_batch = data_batch.withColumn("timestamp",lit(current_timestamp()))
# data_batch = data_batch.withColumn("unix_ts",lit(unix_timestamp('timestamp')))

data_batch = data_batch.withColumn("Id", expr("uuid()"))

# display(data_batch)
data_batch.show(
    3
)  # IMPORTANT AS THIS ENFORCES THE COMPUTATION (e.g. forces the lazy computation to happen now)

# COMMAND ----------

# %sql
# CREATE SCHEMA IF NOT EXISTS iris_data;

# COMMAND ----------

# %sql
# DROP TABLE iris_data.raw_data;
# DROP TABLE iris_data.labels;

# COMMAND ----------

# raw_data_batch = data_batch.drop('target')
# label_batch = data_batch.select('Id','hour','target')
# display(raw_data_batch)
# display(label_batch)

# raw_data_batch.write.format("delta").mode("append").saveAsTable("iris_data.raw_data")
# label_batch.write.format("delta").mode("append").saveAsTable("iris_data.labels")

raw_data_batch = data_batch.drop("target")
label_batch = data_batch.select("Id", "hour", "target")
display(raw_data_batch)
display(label_batch)




# COMMAND ----------

# raw_data_batch.write.option("header", "true").format("delta").mode("append").save(cwd_dev + "raw_data")
# label_batch.write.option("header", "true").format("delta").mode("append").save(cwd_dev + "labels")

# output:
#   database: "default"
#   table: "sklearn_housing"

# db = self.conf["output"].get("database", "default")
# raw_data_table = self.conf["output"]["raw_data_table"]
# label_table = self.conf["output"]["label_table"]
db = "default"
raw_data_table = "raw_data_table"
label_table = "label_table"

raw_data_batch.write.format("delta").mode("overwrite").saveAsTable(f"{db}.{raw_data_table}")
label_batch.write.format("delta").mode("overwrite").saveAsTable(f"{db}.{label_table}")

# COMMAND ----------


