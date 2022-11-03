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


class MonitoringTask(Task):

    # Custom function
    def _monitoring(self, **kwargs):
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
        train_dataset = input_conf["train_dataset"]  
        test_dataset = input_conf["test_dataset"]  
        scored_inference_dataset = input_conf["scored_inference_dataset"]                  

        # Output
        output_conf = self.conf["data"]["output"]
        self.logger.info("output configs: {0}".format(output_conf))       

        db_out = output_conf["database"] 

        # Model configs
        model_conf = self.conf["model"]
        self.logger.info("model configs: {0}".format(model_conf))  
 
        model_name = model_conf["model_name"] 
        experiment = model_conf["experiment_name"] 
        minimal_threshold = model_conf["minimal_threshold"] 
        mlflow.set_experiment(experiment) # Define the MLFlow experiment location

        # =============================
        # 1. Loading the data
        # =============================

        # check this
        listing = self.dbutils.fs.ls("dbfs:/")
        for l in listing:
            self.logger.info(f"DBFS directory: {l}")   

        try:

            # Extract the right version of the training dataset (as logged in MLflow)
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(latest_model.run_id)
            train_dataset_version = run.data.tags['train_dataset_version']
            test_dataset_version = run.data.tags['test_dataset_version']

            train_dataset = spark.read.option("versionAsOf", train_dataset_version).table(f"{db_in}.{train_dataset}")
            test_dataset = spark.read.option("versionAsOf", test_dataset_version).table(f"{db_in}.{test_dataset}")
            scored_inference_dataset = spark.table(f"{db_in}.{scored_inference_dataset}")                        

            self.logger.info("Step 1. completed: Loading the data")   
          
        except Exception as e:
            print("Errored on 1.: Loading the data")
            print("Exception Trace: {0}".format(e))
            # print(traceback.format_exc())
            raise e    

        # ========================================
        # 2. Data monitoring
        # ========================================  

        try:       

            train_dataset_pd = train_dataset.toPandas()
            train_dataset_pd.drop('target', inplace=True, axis=1)
            
            # Data drift calculation
            data_columns = ColumnMapping()
            data_columns.numerical_features = train_dataset_pd.columns #['sl_norm', 'sw_norm', 'pl_norm', 'pw_norm']
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])
            df_with_predictions_pd = df_with_predictions.toPandas()
            print(train_dataset_pd.columns)
            print(df_with_predictions_pd.columns)
            data_drift_profile.calculate(train_dataset_pd, df_with_predictions_pd, column_mapping=data_columns) 
            data_drift_profile_dict = json.loads(data_drift_profile.json())
            print(data_drift_profile.json())
            print(data_drift_profile_dict['data_drift'])
            
            # Save the data monitoring to data lake 
            data_monitor_json = json.dumps(data_drift_profile_dict['data_drift'])
            data_monitor_df = spark.read.json(sc.parallelize([data_monitor_json]))
            display(data_monitor_df)
            data_monitor_df.write.option("header", "true").format("delta").mode("overwrite").save(cwd+"data_monitoring")

            self.logger.info("Step 2 completed: data monitoring")  

        except Exception as e:
            print("Errored on step 2: data monitoring")
            print("Exception Trace: {0}".format(e))
            print(traceback.format_exc())
            raise e        


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
        self.logger.info("Launching monitoring task")
        self._monitoring()
        self.logger.info("Monitoring task finished!")  

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = MonitoringTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
