from databricks_poc_template.common import Task
from databricks_poc_template import module

import mlflow


class TransitionToProdTask(Task):

    # Custom function
    def _transition_to_prod(self, **kwargs):
        # ===========================
        # 0. Reading the config files
        # ===========================

        # Environment
        env = self.conf["environment"]
        self.logger.info("environment: {0}".format(env))

        # Input        

        # Output

        # Model configs
        model_conf = self.conf["model"]
        self.logger.info("model configs: {0}".format(model_conf))  
 
        model_name = model_conf["model_name"] 
        experiment = model_conf["experiment_name"] 
        mlflow.set_experiment(experiment) # Define the MLFlow experiment location    

        # ========================================
        # 1.0 Model transition to prod
        # ========================================
        try:        
            # Initialize client
            client = mlflow.tracking.MlflowClient()
            model_names = [m.name for m in client.list_registered_models()]
            print(model_names)

            # Extracting model information
            mv = client.get_latest_versions(model_name, ['Staging'])[0]
            version = mv.version
            run_id = mv.run_id
            artifact_uri = client.get_model_version_download_uri(model_name, version)
            print(version, artifact_uri, run_id)

            # Model transition to prod
            client.transition_model_version_stage(name=model_name, version=version, stage="Production")
                                    
            # print("Step 1.0 completed: model transition to prod")  
            self.logger.info("Step 1.0 completed: model transition to prod")                

        except Exception as e:
            print("Errored on step 1.0: model transition to prod")
            print("Exception Trace: {0}".format(e))
            print(traceback.format_exc())
            raise e                          

    def launch(self):
        self.logger.info("Launching transition to prod task")
        self._transition_to_prod()
        self.logger.info("Transition to prod task finished!")  

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = TransitionToProdTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
