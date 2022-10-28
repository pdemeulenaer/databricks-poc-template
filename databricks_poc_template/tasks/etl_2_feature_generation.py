from databricks_poc_template.common import Task
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyspark.sql.functions import *
from databricks import feature_store
from databricks_poc_template import module


class ETL2FeatureGenerationTask(Task):
    def _generate_features(self):
        # ===========================
        # 0. Reading the config files
        # ===========================

        # Input
        db = self.conf["input"].get("database", "default")    
        raw_data_table = self.conf["input"]["raw_data_table"]    

        # Output
        fs_schema = self.conf["output"]["fs_schema"] 
        fs_table = self.conf["output"]["fs_table"]             

        # =======================
        # 1. Loading the raw data
        # =======================

        raw_data_batch = spark.table(f"{db}.{raw_data_table}")
        raw_data_batch.show(5)

        # ========================================
        # 1. Feature generation for the data batch
        # ========================================        

        features_df = module.scaled_features_fn(spark, raw_data_batch)
        features_df.show()

        # ===========================================
        # 2. Load the features into the Feature Store
        # ===========================================  

        self.logger.info(f"Writing data batch to feature table {fs_schema}.{fs_table}")

        # reduce partitions to avoid small files problem
        spark.conf.set("spark.sql.shuffle.partitions", "5")

        # Initialize the Feature Store client (feature store in local workspace)
        fs = feature_store.FeatureStoreClient()

        # Make sure that the schema exists
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {fs_schema}") 

        # If the feature table does not exists, create it
        if not spark.catalog._jcatalog.tableExists(f"{fs_schema}.{fs_table}"):
            print("Created feature table: ", f"{fs_schema}.{fs_table}")
            fs.create_table(
                name=f"{fs_schema}.{fs_table}",
                primary_keys=["Id", "hour"],
                df=features_df,
                partition_columns="date",
                description="Iris scaled Features",
            )
        else:
            # Update the feature store table (update only specific rows)
            print("Updated feature table: ", f"{fs_schema}.{fs_table}")
            fs.write_table(
                name=f"{fs_schema}.{fs_table}",
                df=features_df,
                mode="merge",
            )               

        self.logger.info("Dataset successfully written to the Feature Store")

    def launch(self):
        self.logger.info("Launching ETL 2: Feature Generation task")
        self._generate_features()
        self.logger.info("ETL 2: Feature Generation task finished!")

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = ETL2FeatureGenerationTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
