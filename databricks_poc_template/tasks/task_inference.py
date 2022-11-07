from databricks_poc_template.common import Task
from databricks_poc_template import module

# General packages
import pandas as pd
import numpy as np
import json
from pyspark.sql.functions import *

# Databricks
import mlflow
from databricks import feature_store
from mlflow.tracking import MlflowClient

# Monitoring
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, ClassificationPerformanceProfileSection
from evidently.pipeline.column_mapping import ColumnMapping


class InferenceTask(Task):

    # Custom function
    def _inference(self, **kwargs):
        # ===========================
        # 0. Reading the config files
        # ===========================

        # Environment
        env = self.conf["environment"]
        self.logger.info("environment: {0}".format(env))

        # Input
        input_conf = self.conf["data"]["input"]
        self.logger.info("input configs: {0}".format(input_conf))

        db_in = input_conf["database"]  
        inference_dataset = input_conf["inference_dataset"]            

        # Output
        output_conf = self.conf["data"]["output"]
        self.logger.info("output configs: {0}".format(output_conf))       

        db_out = output_conf["database"] 
        scored_inference_dataset = output_conf["scored_inference_dataset"]

        # Model configs
        model_conf = self.conf["model"]
        self.logger.info("model configs: {0}".format(model_conf))  
 
        model_name = model_conf["model_name"] 
        experiment = model_conf["experiment_name"] 
        minimal_threshold = model_conf["minimal_threshold"] 
        mlflow.set_experiment(experiment) # Define the MLFlow experiment location

        # =============================
        # 1. Loading the Inference data
        # =============================

        # check this
        listing = self.dbutils.fs.ls("dbfs:/")
        for l in listing:
            self.logger.info(f"DBFS directory: {l}")   

        try:
            print()
            print("-----------------------------------")
            print("         Model Inference           ")
            print("-----------------------------------")
            print()

            # Load the inference raw data
            raw_data = spark.table(f"{db_in}.{inference_dataset}") # Should we load all of it, every time???

            # max_date = raw_data.select("date").rdd.max()[0]
            # print(max_date)
            # raw_data = raw_data.withColumn("filter_out", when(col("date")==max_date,"1").otherwise(0)) # don't take last day of data
            # raw_data = raw_data.filter("filter_out==1").drop("filter_out")
            # display(raw_data)

            self.logger.info("Step 1. completed: Loading the Inference data")   
          
        except Exception as e:
            print("Errored on 1.: Loading the Inference data")
            print("Exception Trace: {0}".format(e))
            # print(traceback.format_exc())
            raise e    

        # ========================================
        # 2. Model inference
        # ========================================
        try:   
            # Initialize the Feature Store client
            fs = feature_store.FeatureStoreClient()

            # Get the model URI
            latest_model = module.get_latest_model_version(model_name,"databricks")
            latest_model_version = int(latest_model.version)
            model_uri = f"models:/" + model_name + f"/{latest_model_version}"

            print(model_uri)
            raw_data.show(5,False)

            # Call score_batch to get the predictions from the model
            # raw_data = raw_data.select('sepal_length', 'sepal_width', 'petal_length', 'petal_width') # FIXME:
            df_with_predictions = fs.score_batch(model_uri, raw_data) # FIXME: this should be MODIFIED. Follow Joshua way of doing !!!
            df_with_predictions.show(5,False)
            display(df_with_predictions)    

            # Write scored data
            df_with_predictions.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{db_out}.{scored_inference_dataset}")                             

            # print("Step 2. completed: model inference")  
            self.logger.info("Step 2. completed: model inference")                

        except Exception as e:
            print("Errored on step 2.: model inference")
            print("Exception Trace: {0}".format(e))
            print(traceback.format_exc())
            raise e         

    def launch(self):
        self.logger.info("Launching inference task")
        self._inference()
        self.logger.info("Inference task finished!")  

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = InferenceTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
