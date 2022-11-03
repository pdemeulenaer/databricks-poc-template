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
        label_table = input_conf["label_table"]                

        # Output
        output_conf = self.conf["data"]["output"]
        self.logger.info("output configs: {0}".format(output_conf))       

        db_out = output_conf["database"] 
        data_monitoring = output_conf["data_monitoring"] 
        performance_monitoring = output_conf["performance_monitoring"] 

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
            latest_model = module.get_latest_model_version(model_name,"databricks")
            latest_model_version = int(latest_model.version)
            model_uri = f"models:/" + model_name + f"/{latest_model_version}"
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(latest_model.run_id)

            train_dataset_version = run.data.tags['train_dataset_version']
            test_dataset_version = run.data.tags['test_dataset_version']

            train_dataset = spark.read.option("versionAsOf", train_dataset_version).table(f"{db_in}.{train_dataset}")
            test_dataset = spark.read.option("versionAsOf", test_dataset_version).table(f"{db_in}.{test_dataset}")
            df_with_predictions = spark.table(f"{db_in}.{scored_inference_dataset}")   
            label_table =  spark.table(f"{db_in}.{label_table}")                       

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
            data_monitoring_json = json.dumps(data_drift_profile_dict['data_drift'])
            data_monitoring_df = spark.read.json(sc.parallelize([data_monitoring_json]))
            display(data_monitoring_df)
            data_monitoring_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{db_out}.{data_monitoring}") 

            self.logger.info("Step 2 completed: data monitoring")  

        except Exception as e:
            print("Errored on step 2: data monitoring")
            print("Exception Trace: {0}".format(e))
            print(traceback.format_exc())
            raise e        

        # # ========================================
        # # 3. Performance monitoring  (Here assumption of no delayed outcome!)
        # # ========================================   
        # try:        

        #     test_dataset_pd = test_dataset.toPandas()

        #     # Load the target labels of the unseen data (the ones we tried to infer in step 1.1). Here is the assumption of no delayed outcome... 
        #     df_with_predictions = df_with_predictions.join(label_table, ['Id','hour'])
            
        #     # Performance drift calculation
        #     data_columns = ColumnMapping()
        #     data_columns.target = 'target'
        #     data_columns.prediction = 'prediction'
        #     list_columns = [item for item in test_dataset_pd.columns if item not in ['target','prediction']]
        #     data_columns.numerical_features = list_columns #['sl_norm', 'sw_norm', 'pl_norm', 'pw_norm']

        #     performance_drift_profile = Profile(sections=[ClassificationPerformanceProfileSection()])
        #     df_with_predictions_pd = df_with_predictions.toPandas()
        #     print(test_dataset_pd.columns)
        #     print(df_with_predictions_pd.columns)
        #     print(test_dataset_pd.head())
        #     print(df_with_predictions_pd.head())
        #     performance_drift_profile.calculate(test_dataset_pd, df_with_predictions_pd, column_mapping=data_columns) 
        #     performance_drift_profile_dict = json.loads(performance_drift_profile.json())
        #     print(performance_drift_profile.json())
        #     print(performance_drift_profile_dict)
            
        #     # Save the data monitoring to data lake 
        #     performance_monitoring_json = json.dumps(performance_drift_profile_dict)
        #     performance_monitoring_df = spark.read.json(sc.parallelize([performance_monitoring_json]))
        #     print(performance_monitoring_df)
        #     performance_monitoring_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{db_out}.{performance_monitoring}") 

        #     self.logger.info("Step 3 completed: performance monitoring")  

        # except Exception as e:
        #     print("Errored on step 3: performance monitoring")
        #     print("Exception Trace: {0}".format(e))
        #     print(traceback.format_exc())
        #     raise e    
       
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
