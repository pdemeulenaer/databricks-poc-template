# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Sample notebook

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Aux steps for auto reloading of dependent files

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Example usage of existing code

# COMMAND ----------

from databricks_poc_template.tasks.sample_ml_task import SampleMLTask

pipeline = SampleMLTask.get_pipeline()
print(pipeline)

# COMMAND ----------



# COMMAND ----------

import numpy as np
import pandas as pd
import random
import uuid
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql.functions import *
from databricks import feature_store
from databricks_poc_template import module

# COMMAND ----------

# Initialize the dataframe
iris = load_iris()
iris_generated_all = pd.DataFrame(columns = iris.feature_names)

# Generate 50 sample randomly out of each target class
for target_class in [0,1,2]:
  iris_generated = module.iris_data_generator(target_class=str(target_class),n_samples=50) #module.iris_data_generator...
  iris_generated_all = pd.concat([iris_generated_all, iris_generated], axis=0, ignore_index=True)

data_batch = spark.createDataFrame(iris_generated_all)
data_batch = data_batch.withColumnRenamed('sepal length (cm)','sepal_length')
data_batch = data_batch.withColumnRenamed('sepal width (cm)','sepal_width',)
data_batch = data_batch.withColumnRenamed('petal length (cm)','petal_length')
data_batch = data_batch.withColumnRenamed('petal width (cm)','petal_width',)

# data_batch = data_batch.withColumn('Id', monotonically_increasing_id())
# data_batch = data_batch.withColumn('month',lit('202203'))
data_batch = data_batch.withColumn('date',current_date())
data_batch = data_batch.withColumn('hour',hour(current_timestamp()))
# data_batch = data_batch.withColumn("timestamp",lit(current_timestamp()))
# data_batch = data_batch.withColumn("unix_ts",lit(unix_timestamp('timestamp')))

data_batch = data_batch.withColumn("Id",expr("uuid()"))

# display(data_batch)
data_batch.show(3) 

# COMMAND ----------

def iris_data_generator(target_class='all',n_samples=10):
  '''
  This function is meant to generate random samples from a PDF fitted on Iris dataset using Bayesian GMM
  Input:
    - target_class: the desired target class to be generated. Options:
      - '0': for class 0
      - '1': for class 1
      - '2': for class 2
      - 'all': for a random mix of all classes (not available yet)
    - n_samples: the desired number of samples generated
  Output:
    - final_data_generated: the dataframe containing the generated samples (including the target label)
  '''

  # Loading the iris dataset
  iris = datasets.load_iris()
  iris_df = pd.DataFrame(iris.data,columns = iris.feature_names)
  iris_df['target'] = iris.target

  # Initialize the output dataframe
  final_data_generated = pd.DataFrame(columns = iris.feature_names)

  # Selecting the desired target class
  if target_class=='0': weights_target_class=[1,0,0]
  elif target_class=='1': weights_target_class=[0,1,0]
  elif target_class=='2': weights_target_class=[0,0,1]
  else: weights_target_class=[1./3.,1./3.,1./3.]
  
  # Now we need to generate samples for each of the 3 classes
  samples_per_class = random.choices([0,1,2], weights=weights_target_class, k=n_samples)

  # Target class id and counts per target class:
  class_id, counts_per_class = np.unique(samples_per_class, return_counts=True)

  # Looping on the 3 target classes
  for j,one_class_id in enumerate(class_id):

    # Extract the data of a given target class
    subset_df = iris_df[iris_df['target']==one_class_id]
    subset_df.drop('target', axis=1, inplace=True)

    # Fit the Bayesian GMM on the data
    n_components = 10 # Number of Gaussian components in the GMM model
    gmm = BayesianGaussianMixture(n_components=n_components,
                                  covariance_type='full', 
                                  # tol=0.00001, 
                                  # reg_covar=1e-06, 
                                  max_iter=20, 
                                  random_state=0, 
                                  n_init=10,
                                  # weight_concentration_prior=0.1
                                  )

    gmm.fit(subset_df.to_numpy()) 

    means = gmm.means_
    cov = gmm.covariances_
    weights = gmm.weights_

    # Compute the number of samples for each component of the GMM PDF
    # Indeed the GMM pdf is made of multiple Gaussian components.
    # So we sample each component respecting its own weight
    # The "counts" list is a list of the number of samples for each component
    component_samples = random.choices(population=np.arange(n_components),  # list to pick from
                                      weights=weights,  # weights of the population, in order
                                      k=counts_per_class[j]  # amount of samples to draw
                                      )
    # print(component_samples)

    component_id, counts_per_component = np.unique(component_samples, return_counts=True)
    # print(component_id, counts_per_component)

    # Generate the samples for each GMM components following the counts
    data_gen = np.random.multivariate_normal(means[component_id[0],:],cov[component_id[0]],counts_per_component[0]) 
    for i in range(1,len(component_id)):
      data_new = np.random.multivariate_normal(means[component_id[i],:],cov[component_id[i]],counts_per_component[i]) 
      data_gen = np.vstack((data_gen,data_new)) 
      del data_new 

    data_generated_per_class = pd.DataFrame(data_gen,columns = iris.feature_names)
    data_generated_per_class['target'] = one_class_id

    final_data_generated = pd.concat([final_data_generated, data_generated_per_class], axis=0, ignore_index=True)

  return final_data_generated

# COMMAND ----------

from sklearn.datasets import make_multilabel_classification
X, Y = make_multilabel_classification(
    n_classes=3, 
    n_labels=1, 
    n_samples=5,
    n_features=3,
    allow_unlabeled=False, 
    random_state=42
)

X, Y

# COMMAND ----------

from sklearn.datasets import make_blobs
centers = [(-2, -2, -2), (0, 0, 0), (2, 2, 2)]
data, y = make_blobs(n_samples=100,
                     n_features=3, 
                     centers=centers, 
                     shuffle=False, 
                     random_state=42)
data,y

# COMMAND ----------

# Initialize plot
fig, axs = plt.subplots(2, 3, figsize=(17,10))

colors_samples = ['blue','red','green']
colors_iris = ['darkblue','darkred','darkgreen']
colors = ['navy', 'turquoise', 'darkorange','navy', 'turquoise', 'darkorange','navy', 'turquoise', 'darkorange']
markers = ['+','*','x','+','*','x','+','*','x','+','*','x']

# for k in [0,1,2]:
#   data = iris.data[iris.target == k]
#   target_k = iris_generated['target'] == k

axs[0, 0].scatter(data[:, 0], data[:, 1],alpha=0.5)
#   axs[0, 0].set_xlim([0.5,7.5])
#   axs[0, 0].set_ylim([-0.2,2.75])
#   axs[0, 0].set_ylabel('petal width (cm)',size=16)
#   axs[0, 0].set_xticklabels([])

axs[0, 1].scatter(data[:, 0], data[:, 2],alpha=0.5)
#   axs[0, 1].set_xlim([1.5,4.75])
#   axs[0, 1].set_ylim([0.8,7.8])
#   axs[0, 1].set_ylabel('petal length (cm)',size=16)
#   axs[0, 1].set_xticklabels([])

axs[0, 2].scatter(data[:, 1], data[:, 2],alpha=0.5)
#   axs[0, 1].set_xlim([1.5,4.75])
#   axs[0, 1].set_ylim([0.8,7.8])
#   axs[0, 1].set_ylabel('petal length (cm)',size=16)
#   axs[0, 1].set_xticklabels([])

plt.show()

# COMMAND ----------


