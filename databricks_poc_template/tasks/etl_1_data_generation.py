from databricks_poc_template.common import Task
import numpy as np
import pandas as pd
import random
import uuid
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql.functions import *
from databricks_poc_template import module


class ETL1DataGenerationTask(Task):
    def _generate_data(self):
        # ===========================
        # 0. Reading the config files
        # ===========================

        # Output
        db = self.conf["output"].get("database", "default")
        raw_data_table = self.conf["output"]["raw_data_table"]
        label_table = self.conf["output"]["label_table"]

        # ===============================
        # 1. Creation of a new data batch
        # ===============================        

        # Initialize the dataframe
        iris = load_iris()
        iris_generated_all = pd.DataFrame(columns=iris.feature_names)

        # Generate 50 sample randomly out of each target class
        for target_class in [0, 1, 2]:
            iris_generated = module.iris_data_generator(
                target_class=str(target_class), 
                n_samples=50
            )  
            iris_generated_all = pd.concat(
                [iris_generated_all, iris_generated], axis=0, ignore_index=True
            )

        data_batch = spark.createDataFrame(iris_generated_all)
        data_batch = data_batch.withColumnRenamed("sepal length (cm)", "sepal_length")
        data_batch = data_batch.withColumnRenamed("sepal width (cm)", "sepal_width")
        data_batch = data_batch.withColumnRenamed("petal length (cm)", "petal_length")
        data_batch = data_batch.withColumnRenamed("petal width (cm)", "petal_width")

        # data_batch = data_batch.withColumn('Id', monotonically_increasing_id())
        # data_batch = data_batch.withColumn('month',lit('202203'))
        data_batch = data_batch.withColumn("date", current_date())
        data_batch = data_batch.withColumn("hour", hour(current_timestamp()))
        # data_batch = data_batch.withColumn("timestamp",lit(current_timestamp()))
        # data_batch = data_batch.withColumn("unix_ts",lit(unix_timestamp('timestamp')))

        data_batch = data_batch.withColumn("Id", expr("uuid()"))

        raw_data_batch = data_batch.drop("target")
        label_batch = data_batch.select("Id", "date", "hour", "target")
        display(raw_data_batch)
        display(label_batch)

        # %sql
        # CREATE SCHEMA IF NOT EXISTS iris_data;

        # %sql
        # DROP TABLE iris_data.raw_data;
        # DROP TABLE iris_data.labels;

        # ====================================================
        # 2. Write the data batch to a Table in Hive Metastore
        # ====================================================        
        self.logger.info(f"Writing data batch to {db}.{table}")
        raw_data_batch.write.format("delta").mode("append").option("overwriteSchema", "true").saveAsTable(f"{db}.{raw_data_table}")
        label_batch.write.format("delta").mode("append").option("overwriteSchema", "true").saveAsTable(f"{db}.{label_table}")
        self.logger.info("Dataset successfully written")

    def launch(self):
        self.logger.info("Launching ETL 1: Data Generation task")
        self._generate_data()
        self.logger.info("ETL 1: Data Generation task finished!")

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = ETL1DataGenerationTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
