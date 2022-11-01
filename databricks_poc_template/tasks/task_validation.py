from databricks_poc_template.common import Task
from databricks_poc_template import module

import pandas as pd
import numpy as np
import mlflow
import json

# Import of Sklearn packages
from sklearn.metrics import accuracy_score, confusion_matrix

# Import matplotlib packages
# from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
# import pylab
from pylab import *
# import matplotlib.cm as cm
# import matplotlib.mlab as mlab


class ValidationTask(Task):

    # Custom function
    def _validate(self, **kwargs):
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
        test_dataset = input_conf["test_dataset"]            

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

        # ========================
        # 1. Loading the Test data
        # ========================

        # check this
        listing = self.dbutils.fs.ls("dbfs:/")
        for l in listing:
            self.logger.info(f"DBFS directory: {l}")   

        try:
            # Load the raw data and associated label tables
            test_df = spark.table(f"{db_in}.{test_dataset}")
            test_pd = test_df.toPandas()

            # Feature selection # TODO: AUTOMATE THIS!!!
            feature_cols = ["sl_norm","sw_norm","pl_norm","pw_norm"] #['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            target = 'target'   

            X_test = test_pd[feature_cols]
            y_test = test_pd[target].values            
    
            # print("Step 1. completed: Loaded Test data")   
            self.logger.info("Step 1. completed: Loaded Test data")   
          
        except Exception as e:
            print("Errored on 1.: data loading")
            print("Exception Trace: {0}".format(e))
            # print(traceback.format_exc())
            raise e    

        # ========================================
        # 2. Load model from MLflow Model Registry
        # ========================================
        try:   
            # Load model from MLflow experiment
            # Conditions:
            # - model accuracy should be higher than pre-defined threshold (defined in model.json)

            # Initialize MLflow client
            client = mlflow.tracking.MlflowClient()
            model_names = [m.name for m in client.search_registered_models()]
            print(model_names)

            # Extracting model & its information (latest model with tag 'None')
            mv = client.get_latest_versions(model_name, ['None'])[0]
            version = mv.version
            run_id = mv.run_id
            artifact_uri = client.get_model_version_download_uri(model_name, version)
            model = mlflow.pyfunc.load_model(artifact_uri)            

            # print("Step 2. completed: load model from MLflow")  
            self.logger.info("Step 2. completed: load model from MLflow")                

        except Exception as e:
            print("Errored on step 2.: model loading from MLflow")
            print("Exception Trace: {0}".format(e))
            print(traceback.format_exc())
            raise e      

        # =============================================================
        # 3. Model validation (and tagging "staging") in Model Registry
        # =============================================================
        try:                      
            # Derive accuracy on TEST dataset
            y_test_pred = model.predict(X_test) 
            test_pd['prediction'] = y_test_pred
            test_df_out = spark.createDataFrame(test_pd)            
            test_df_out.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{db_out}.{test_dataset}")             

            # Accuracy and Confusion Matrix
            test_accuracy = accuracy_score(y_test, y_test_pred)
            print('TEST accuracy = ',test_accuracy)
            print('TEST Confusion matrix:')
            Classes = ['setosa','versicolor','virginica']
            C = confusion_matrix(y_test, y_test_pred)
            C_normalized = C / C.astype(np.float).sum()        
            C_normalized_pd = pd.DataFrame(C_normalized,columns=Classes,index=Classes)
            print(C_normalized_pd)   

            # Figure plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(C,cmap='Blues')
            plt.title('Confusion matrix of the classifier')
            fig.colorbar(cax)
            ax.set_xticklabels([''] + Classes)
            ax.set_yticklabels([''] + Classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig("confusion_matrix_TEST.png")    

            with mlflow.start_run(run_id) as run:

                # Tracking performance metrics on TEST dataset   
                mlflow.log_metric("accuracy_TEST", test_accuracy)
                mlflow.log_figure(fig, "confusion_matrix_TEST.png")  

                # IF we pass the validation, we push the model to Staging tag 
                print(f"Minimal accuracy threshold: {minimal_threshold:5.2f}")          
                if test_accuracy >= minimal_threshold: 
                    mlflow.set_tag("validation", "passed")
                    if env == 'staging': 
                        client.transition_model_version_stage(name=model_name, version=version, stage="Staging")
                else: 
                    mlflow.set_tag("validation", "failed")

                # Tracking the Test dataset (with predictions)
                test_dataset_version = module.get_table_version(spark,f"{db_out}.{test_dataset}")
                mlflow.set_tag("test_dataset_version", test_dataset_version)
                mlflow.set_tag("test_dataset", f"{db_out}.{test_dataset}")                    
                            
            # print("Step 3. completed: model validation")  
            self.logger.info("Step 3. completed: model validation")                

        except Exception as e:
            print("Errored on step 3.: model validation")
            print("Exception Trace: {0}".format(e))
            print(traceback.format_exc())
            raise e                       

    def launch(self):
        self.logger.info("Launching validation task")
        self._validate()
        self.logger.info("Validation task finished!")  

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = ValidationTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
