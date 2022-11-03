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
            latest_model = module.get_latest_model_version(model_name,registry_uri)
            latest_model_version = int(latest_model.version)
            model_uri = f"models:/" + model_name + f"/{latest_model_version}"

            # Call score_batch to get the predictions from the model
            df_with_predictions = fs.score_batch(model_uri, raw_data)
            display(df_with_predictions)    

            # Write scored data
            # df_with_predictions.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{db_out}.{scored_inference_dataset}")                             

            # print("Step 2. completed: model inference")  
            self.logger.info("Step 2. completed: model inference")                

        except Exception as e:
            print("Errored on step 2.: model inference")
            print("Exception Trace: {0}".format(e))
            print(traceback.format_exc())
            raise e

        # # ========================================
        # # 3. Data monitoring
        # # ========================================  

        # try:       
        #     # Extract the right version of the training dataset (as logged in MLflow)
        #     client = mlflow.tracking.MlflowClient()
        #     run = client.get_run(latest_model.run_id)
        #     train_dataset_version = run.data.tags['train_dataset_version']
        #     train_dataset_path = run.data.tags['train_dataset_path']
        #     # test_dataset_version = run.data.tags['test_dataset_version']
        #     # fs_table_version = run.data.tags['fs_table_version']
        #     train_dataset = spark.read.format("delta").option("versionAsOf", train_dataset_version).load(train_dataset_path)
        #     train_dataset_pd = train_dataset.toPandas()

        #     train_dataset_pd.drop('target', inplace=True, axis=1)
            
        #     # Data drift calculation
        #     data_columns = ColumnMapping()
        #     data_columns.numerical_features = train_dataset_pd.columns #['sl_norm', 'sw_norm', 'pl_norm', 'pw_norm']
        #     data_drift_profile = Profile(sections=[DataDriftProfileSection()])
        #     df_with_predictions_pd = df_with_predictions.toPandas()
        #     print(train_dataset_pd.columns)
        #     print(df_with_predictions_pd.columns)
        #     data_drift_profile.calculate(train_dataset_pd, df_with_predictions_pd, column_mapping=data_columns) 
        #     data_drift_profile_dict = json.loads(data_drift_profile.json())
        #     print(data_drift_profile.json())
        #     print(data_drift_profile_dict['data_drift'])
            
        #     # Save the data monitoring to data lake 
        #     data_monitor_json = json.dumps(data_drift_profile_dict['data_drift'])
        #     data_monitor_df = spark.read.json(sc.parallelize([data_monitor_json]))
        #     display(data_monitor_df)
        #     data_monitor_df.write.option("header", "true").format("delta").mode("overwrite").save(cwd+"data_monitoring")

        #     self.logger.info("Step 1.2 completed: data monitoring")  

        # except Exception as e:
        #     print("Errored on step 1.2: data monitoring")
        #     print("Exception Trace: {0}".format(e))
        #     print(traceback.format_exc())
        #     raise e        


        # # try:
        # # ========================================
        # # 1.3 Performance monitoring  (Here assumption of no delayed outcome!)
        # # ========================================            

        # # Extract the right version of the training dataset (as logged in MLflow)
        # client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
        # run = client.get_run(latest_model.run_id)
        # train_dataset_version = run.data.tags['train_dataset_version']
        # train_dataset_path = run.data.tags['train_dataset_path']
        # # test_dataset_version = run.data.tags['test_dataset_version']
        # # fs_table_version = run.data.tags['fs_table_version']
        # train_dataset = spark.read.format("delta").option("versionAsOf", train_dataset_version).load(train_dataset_path)
        # train_dataset_pd = train_dataset.toPandas()

        # # Load the target labels of the unseen data (the ones we tried to infer in step 1.1). Here is the assumption of no delayed outcome... 
        # labels = spark.read.format('delta').load(cwd + 'labels')
        # df_with_predictions = df_with_predictions.join(labels, ['Id','hour'])
        
        # # Performance drift calculation
        # data_columns = ColumnMapping()
        # data_columns.target = 'target'
        # data_columns.prediction = 'prediction'
        # data_columns.numerical_features = train_dataset_pd.columns #['sl_norm', 'sw_norm', 'pl_norm', 'pw_norm']

        # performance_drift_profile = Profile(sections=[ClassificationPerformanceProfileSection()])
        # df_with_predictions_pd = df_with_predictions.toPandas()
        # print(train_dataset_pd.columns)
        # print(df_with_predictions_pd.columns)
        # print(train_dataset_pd.head())
        # print(df_with_predictions_pd.head())
        # performance_drift_profile.calculate(train_dataset_pd, df_with_predictions_pd, column_mapping=data_columns) 
        # performance_drift_profile_dict = json.loads(performance_drift_profile.json())
        # print(performance_drift_profile.json())
        # print(performance_drift_profile_dict)
        
        # # Save the data monitoring to data lake 
        # performance_monitor_json = json.dumps(performance_drift_profile_dict)
        # performance_monitor_df = spark.read.json(sc.parallelize([performance_monitor_json]))
        # print(performance_monitor_df)
        # performance_monitor_df.write.option("header", "true").format("delta").mode("overwrite").save(cwd+"performance_monitoring")

        # self.logger.info("Step 1.3 completed: performance monitoring")  

        # # except Exception as e:
        # #     print("Errored on step 1.2: data monitoring")
        # #     print("Exception Trace: {0}".format(e))
        # #     print(traceback.format_exc())
        # #     raise e           

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
