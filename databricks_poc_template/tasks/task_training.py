from databricks_poc_template.common import Task
from databricks_poc_template import module

# General packages
import pandas as pd
import numpy as np
import mlflow
import json
from pyspark.sql.functions import *

# Import matplotlib packages
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import pylab
from pylab import *
import matplotlib.cm as cm
import matplotlib.mlab as mlab

# Sklearn packages
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Databricks packages
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.sklearn #mlflow.lightgbm
from mlflow.models.signature import infer_signature
from mlflow.tracking.artifact_utils import get_artifact_uri
from databricks import feature_store
from databricks.feature_store import FeatureLookup


class TrainTask(Task):

    # Custom function
    def _train(self, **kwargs):
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
        raw_data_table = input_conf["raw_data_table"]  
        label_table = input_conf["label_table"] 
        fs_schema = input_conf["fs_schema"] 
        fs_table = input_conf["fs_table"]          

        # Output
        output_conf = self.conf["data"]["output"]
        self.logger.info("output configs: {0}".format(output_conf))       

        db_out = output_conf["database"]   
        train_dataset = output_conf["train_dataset"] 
        test_dataset = output_conf["test_dataset"]         

        # Model configs
        model_conf = self.conf["model"]
        self.logger.info("model configs: {0}".format(model_conf))  
 
        model_name = model_conf["model_name"] 
        experiment = model_conf["experiment_name"] 
        mlflow.set_experiment(experiment) # Define the MLFlow experiment location

        # =======================
        # 1. Loading the raw data
        # =======================

        # check this
        listing = self.dbutils.fs.ls("dbfs:/")
        for l in listing:
            self.logger.info(f"DBFS directory: {l}")           

        try:        
            # Load the raw data and associated label tables
            raw_data = spark.table(f"{db_in}.{raw_data_table}")
            labels = spark.table(f"{db_in}.{label_table}")
            
            # Joining raw_data and labels
            raw_data_with_labels = raw_data.join(labels, ['Id','date','hour'])
            display(raw_data_with_labels)
            
            # Selection of the data and labels until last LARGE time step (e.g. day or week let's say)
            # Hence we will remove the last large timestep of the data
            # max_hour = raw_data_with_labels.select("hour").rdd.max()[0]
            max_date = raw_data_with_labels.select("date").rdd.max()[0]
            print(max_date)
            # raw_data_with_labels = raw_data_with_labels.withColumn("filter_out", when((col("hour")==max_hour) & (col("date")==max_date),"1").otherwise(0)) # don't take last hour of last day of data
            raw_data_with_labels = raw_data_with_labels.withColumn("filter_out", when(col("date")==max_date,"1").otherwise(0)) # don't take last day of data
            raw_data_with_labels = raw_data_with_labels.filter("filter_out==0").drop("filter_out")
            display(raw_data_with_labels)
            
            self.logger.info("Step 1.0 completed: Loaded historical raw data and labels")   
          
        except Exception as e:
            print("Errored on 1.0: data loading")
            print("Exception Trace: {0}".format(e))
            # print(traceback.format_exc())
            raise e  

        # ==================================
        # 2. Building the training dataset
        # ==================================
        try:        
            # Initialize the Feature Store client
            fs = feature_store.FeatureStoreClient()

            # Declaration of the features, in a "feature lookup" object
            feature_lookups = [
                FeatureLookup( 
                table_name = f"{fs_schema}.{fs_table}",
                feature_names = ["sl_norm","sw_norm","pl_norm","pw_norm"], # TODO: automate this in config file
                lookup_key = ["Id","hour"], # TODO: automate this in config file
                ),
            ]

            # Create the training dataset (includes the raw input data merged with corresponding features from feature table)
            exclude_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] # TODO: should I exclude the 'Id', 'hour','date'? 
            training_set = fs.create_training_set(
                df = raw_data_with_labels,
                feature_lookups = feature_lookups,
                label = "target",
                exclude_columns = exclude_columns
            )

            # Load the training dataset into a dataframe
            training_df = training_set.load_df()
            display(training_df)
            training_df.show(5)
        
            # Collect data into a Pandas array for training
            features_and_label = training_df.columns
            data_pd = training_df.toPandas()[features_and_label]

            # Do the train-test split
            train, test = train_test_split(data_pd, train_size=0.7, random_state=123)
            
            # Save train dataset
            # train_pd = pd.DataFrame(data=np.column_stack((x_train,y_train)), columns=features_and_label)
            # train_df = spark.createDataFrame(train_pd)
            # train_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{db_out}.{train_dataset}")
            train_df = spark.createDataFrame(train)
            train_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{db_out}.{train_dataset}")            
            
            # Save test dataset
            # test_pd = pd.DataFrame(data=np.column_stack((x_test,y_test)), columns=features_and_label)
            # test_df = spark.createDataFrame(test_pd)
            test_df = spark.createDataFrame(test)            
            test_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{db_out}.{test_dataset}") 
    
            self.logger.info("Step 2. completed: Building the training dataset")   
          
        except Exception as e:
            print("Errored on 2.: Building the training dataset")
            print("Exception Trace: {0}".format(e))
            # print(traceback.format_exc())
            raise e  

        # ========================================
        # 1.3 Model training
        # ========================================
        try:            
            with mlflow.start_run() as run:    
                mlflow.sklearn.autolog()                      

                print("Active run_id: {}".format(run.info.run_id))
                self.logger.info("Active run_id: {}".format(run.info.run_id))

                # Model definition
                base_estimator = RandomForestClassifier(oob_score = True,
                                                        random_state=21,
                                                        n_jobs=-1)   

                CV_rfc = GridSearchCV(estimator=base_estimator, 
                                    param_grid=model_conf['hyperparameters_grid'],
                                    cv=5)

                # Remove unneeded data
                x_train = train.drop(["target",'Id', 'hour','date'], axis=1)
                y_train = train.target                
                # x_test = test.drop(["target",'Id', 'hour','date'], axis=1)
                # y_test = test.target

                # Cross validation model fit
                CV_rfc.fit(x_train, y_train)
                print(CV_rfc.best_params_)
                print(CV_rfc.best_score_)
                print(CV_rfc.best_estimator_)
                model = CV_rfc.best_estimator_

                # Tracking the model parameters
                train_dataset_version = module.get_table_version(spark,f"{db_out}.{train_dataset}")
                # test_dataset_version = module.get_table_version(spark,f"{db_out}.{test_dataset}") # done in validation step
                fs_table_version = module.get_table_version(spark,f"{fs_schema}.{fs_table}")
                mlflow.set_tag("train_dataset_version", train_dataset_version)
                # mlflow.set_tag("test_dataset_version", test_dataset_version) # done in validation step
                mlflow.set_tag("fs_table_version", fs_table_version)
                mlflow.set_tag("train_dataset", f"{db_out}.{train_dataset}")
                # mlflow.set_tag("test_dataset", f"{db_out}.{test_dataset}") # done in validation step
                mlflow.set_tag("raw_data", f"{db_in}.{raw_data_table}")
                mlflow.set_tag("raw_labels", f"{db_in}.{label_table}")
                mlflow.set_tag("environment run", f"{env}") # Tag the environment where the run is done
                signature = infer_signature(x_train, model.predict(x_train))  

                # Add an random input example for the model
                input_example = {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }                               
                
                # Log the model
                # mlflow.sklearn.log_model(model, "model") #, registered_model_name="sklearn-rf")   

                # Register the model to MLflow MR as well as FS MR (should not register in DEV?)
                fs.log_model(
                    model,
                    artifact_path=model_name,
                    flavor=mlflow.sklearn,
                    training_set=training_set,
                    # registered_model_name=model_name,
                )
                
                # Register the model to the Model Registry
                print(mlflow.get_registry_uri())
                mlflow.sklearn.log_model(model, 
                                        model_name,
                                        registered_model_name=model_name,
                                        signature=signature,
                                        input_example=input_example)           

                self.logger.info("Step 3 completed: model training and saved to MLFlow")                

        except Exception as e:
            print("Errored on step 3: model training")
            print("Exception Trace: {0}".format(e))
            print(traceback.format_exc())
            raise e   
               
        
    def launch(self):
        self.logger.info("Launching train task")
        self._train()
        self.logger.info("Train task finished!")  

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = TrainTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()



