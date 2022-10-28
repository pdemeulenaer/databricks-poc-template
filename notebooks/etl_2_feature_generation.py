# Databricks notebook source
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyspark.sql.functions import *
from databricks import feature_store
from databricks_poc_template import module

# COMMAND ----------

def scaled_features_fn(df):
    """
    Computes the scaled_features feature group.
    To restrict features to a time range, pass in ts_column, start_date, and/or end_date as kwargs.
    """

    pdf = df.toPandas()
    id = pdf["Id"]
    date = pdf["date"]
    hour = pdf["hour"]
    # timestamp = pdf['timestamp']
    # unix_ts = pdf['unix_ts']
    # target = pdf['target']
    pdf.drop("Id", axis=1, inplace=True)
    pdf.drop("date", axis=1, inplace=True)
    pdf.drop("hour", axis=1, inplace=True)
    # pdf.drop('timestamp', axis=1, inplace=True)
    # pdf.drop('unix_ts', axis=1, inplace=True)

    # columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'Id'] #list(pdf.columns)

    # pdf_norm=pdf.to_numpy()
    scaler = StandardScaler()
    scaler.fit(pdf)    
    pdf_norm = scaler.transform(pdf)
    columns = ["sl_norm", "sw_norm", "pl_norm", "pw_norm"]
    pdf_norm = pd.DataFrame(data=pdf_norm, columns=columns)
    # pdf_norm['sl_norm'] = pdf_norm['sl_norm'] * 2
    # pdf_norm['sw_norm'] = pdf_norm['sw_norm'] * 2
    # pdf_norm['pl_norm'] = pdf_norm['pl_norm'] * 2
    # pdf_norm['pw_norm'] = pdf_norm['pw_norm'] * 2
    pdf_norm["Id"] = id
    pdf_norm["date"] = date
    # pdf_norm['timestamp'] = timestamp
    # pdf_norm['unix_ts'] = unix_ts
    pdf_norm["hour"] = hour

    return spark.createDataFrame(pdf_norm)

# COMMAND ----------

raw_data_batch = spark.table("default.raw_data_table")
display(raw_data_batch)

# COMMAND ----------

# Creation of the features
features_df = scaled_features_fn(raw_data_batch)
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


