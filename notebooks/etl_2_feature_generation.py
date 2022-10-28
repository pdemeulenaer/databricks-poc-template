# Databricks notebook source
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyspark.sql.functions import *
from databricks import feature_store
from databricks_poc_template import module

# COMMAND ----------

# Loading of the raw data (for which we want to generate features)
raw_data_batch = spark.table("default.raw_data_table")
display(raw_data_batch)

# COMMAND ----------

# Creation of the features
features_df = module.scaled_features_fn(spark, raw_data_batch)
display(features_df)

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "5")

# Initialize the Feature Store client (feature store in local workspace)
fs = feature_store.FeatureStoreClient()

spark.sql("CREATE SCHEMA IF NOT EXISTS feature_store_poc_template1")

# COMMAND ----------

# Feature store table name
fs_table = "feature_store_poc_template1.scaled_features"

# If the table does not exists, create it
if not spark.catalog._jcatalog.tableExists(fs_table):
    print("Created feature table: ", fs_table)
    fs.create_table(
        name=fs_table,
        primary_keys=["Id", "hour"],
        df=features_df,
        partition_columns="date",
        description="Iris scaled Features",
    )
else:
    # Update the feature store table (update only specific rows)
    print("Updated feature table: ", fs_table)
    fs.write_table(
        name=fs_table,
        df=features_df,
        mode="merge",
    )

# COMMAND ----------


