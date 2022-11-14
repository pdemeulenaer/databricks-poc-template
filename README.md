[![CI pipeline](https://github.com/pdemeulenaer/databricks-poc-template/actions/workflows/ci.yml/badge.svg)](https://github.com/pdemeulenaer/databricks-poc-template/actions/workflows/ci.yml)

## Structure of the Repo

Note: add hidden directories in the tree

```
.
├── conf
│   ├── deployment_original.yml
│   ├── deployment.yml
│   └── tasks
│       ├── etl_1_data_generation.yml
│       ├── etl_2_feature_generation.yml
│       ├── sample_etl_config.yml
│       ├── sample_ml_config.yml
│       ├── task_inference_dev.yml
│       ├── task_inference_prod.yml
│       ├── task_inference_uat.yml
│       ├── task_monitoring_dev.yml
│       ├── task_monitoring_prod.yml
│       ├── task_monitoring_uat.yml
│       ├── task_training_dev.yml
│       ├── task_training_staging.yml
│       ├── task_transition_to_prod.yml
│       ├── task_validation_dev.yml
│       └── task_validation_staging.yml
├── databricks_poc_template
│   ├── common.py
│   ├── __init__.py
│   ├── module.py
│   └── tasks
│       ├── etl_1_data_generation.py
│       ├── etl_2_feature_generation.py
│       ├── __init__.py
│       ├── sample_etl_task.py
│       ├── sample_ml_task.py
│       ├── task_inference.py
│       ├── task_monitoring.py
│       ├── task_training.py
│       ├── task_transition_to_prod.py
│       └── task_validation.py
├── Makefile
├── notebooks
│   ├── etl_1_data_generation.py
│   ├── etl_2_feature_generation.py
│   └── sample_notebook.py
├── pyproject.toml
├── pytest.ini
├── README.md
├── setup.py
├── sonar-project.properties
└── tests
    ├── entrypoint.py
    ├── integration
    │   └── e2e_test.py
    └── unit
        ├── conftest.py
        └── sample_test.py
```

# databricks-poc-template

## Intro

This repository consists of a template that allows to develop and deploy a dummy machine learning model to a target Databricks workspace.

The development and deployment lifecycle span on these environments:

1. Development

2. Staging

3. UAT (or QA, or pre-prod)

4. Production

In this template, we work within a *single* Databricks workspace, so that the different environments are materialized by different Databricks clusters (another, more realistic approach would be to differentiate the environments using multiple Databricks workspaces). While the `Development` environment consists of a personal (often single node) interactive cluster, the other environments are materialized with 3 cluster pools 

## Prerequisites

### 1. How to set up your environment

To be able to connect to the Databricks workspace from the command line, you need first to install the package databricks-cli (https://docs.databricks.com/dev-tools/cli/index.html) in your python environment:

```
pip install databricks-cli
```

Then you can configure the cli, using a PAT, a personal access token that should have been generated beforehand within the Databricks workspace:

```
databricks configure --token
```

Then you have to enter the name of the Databricks workspace (the "host"), and as well enter the token:

```
Databricks Host (should begin with https://):
Token:
```

You then end up with a configuration file, `~/.databrickscfg`, where you have the entered info:

```
[DEFAULT]
host = <workspace-URL>
token = <personal-access-token>
```

where `DEFAULT` is the default profile. We can work with different profiles, associated with different workspaces, but won't use them in the scope of the PoC since we are working within one workspace only. For more info on profiles, https://docs.databricks.com/dev-tools/cli/index.html#connection-profiles

### 2. The template creation with DBX
The creation of Databricks template requires the installation of DBX. Follow https://dbx.readthedocs.io/en/latest/guides/python/python_quickstart/

If you want to create an empty template, you basically need to initiate your project like this:

```
dbx init -p \
    "cicd_tool=GitHub Actions" \
    -p "cloud=AWS" \
    -p "project_name=<your project name>" \
    -p "profile=DEFAULT" \
    --no-input
```

If you want to create a pre-filled template with a use-case and adapted CICD pipeline (including linting, coverage calculation, Sonar code analysis), just fork this repo instead.  
    
## Instructions to reproduce the different steps

## TODO List

- TODO: Generalize the feature list within the config files

- TODO: Allow multiple config files in deployment.yml

- TODO: find way to deploy Classic REST API from MLflow Model Registry


======================

# Initial readme file after this
This is a sample project for Databricks, generated via cookiecutter.

While using this project, you need Python 3.X and `pip` or `conda` for package management.

## Local environment setup

1. Instantiate a local Python environment via a tool of your choice. This example is based on `conda`, but you can use any environment management tool:
```bash
conda create -n databricks_poc_template python=3.9
conda activate databricks_poc_template
```

2. If you don't have JDK installed on your local machine, install it (in this example we use `conda`-based installation):
```bash
conda install -c conda-forge openjdk=11.0.15
```

3. Install project locally (this will also install dev requirements):
```bash
pip install -e ".[local,test]"
```

## Running unit tests

For unit testing, please use `pytest`:
```
pytest tests/unit --cov
```

Please check the directory `tests/unit` for more details on how to use unit tests.
In the `tests/unit/conftest.py` you'll also find useful testing primitives, such as local Spark instance with Delta support, local MLflow and DBUtils fixture.

## Running integration tests

There are two options for running integration tests:

- On an all-purpose cluster via `dbx execute`
- On a job cluster via `dbx launch`

For quicker startup of the job clusters we recommend using instance pools ([AWS](https://docs.databricks.com/clusters/instance-pools/index.html), [Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/instance-pools/), [GCP](https://docs.gcp.databricks.com/clusters/instance-pools/index.html)).

For an integration test on all-purpose cluster, use the following command:
```
dbx execute <workflow-name> --cluster-name=<name of all-purpose cluster>
```

To execute a task inside multitask job, use the following command:
```
dbx execute <workflow-name> \
    --cluster-name=<name of all-purpose cluster> \
    --job=<name of the job to test> \
    --task=<task-key-from-job-definition>
```

For a test on a job cluster, deploy the job assets and then launch a run from them:
```
dbx deploy <workflow-name> --assets-only
dbx launch <workflow-name>  --from-assets --trace
```


## Interactive execution and development on Databricks clusters

1. `dbx` expects that cluster for interactive execution supports `%pip` and `%conda` magic [commands](https://docs.databricks.com/libraries/notebooks-python-libraries.html).
2. Please configure your workflow (and tasks inside it) in `conf/deployment.yml` file.
3. To execute the code interactively, provide either `--cluster-id` or `--cluster-name`.
```bash
dbx execute <workflow-name> \
    --cluster-name="<some-cluster-name>"
```

Multiple users also can use the same cluster for development. Libraries will be isolated per each user execution context.

## Working with notebooks and Repos

To start working with your notebooks from a Repos, do the following steps:

1. Add your git provider token to your user settings in Databricks
2. Add your repository to Repos. This could be done via UI, or via CLI command below:
```bash
databricks repos create --url <your repo URL> --provider <your-provider>
```
This command will create your personal repository under `/Repos/<username>/databricks_poc_template`.
3. Use `git_source` in your job definition as described [here](https://dbx.readthedocs.io/en/latest/examples/notebook_remote.html)

## CI/CD pipeline settings

Please set the following secrets or environment variables for your CI provider:
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`

## Testing and releasing via CI pipeline

- To trigger the CI pipeline, simply push your code to the repository. If CI provider is correctly set, it shall trigger the general testing pipeline
- To trigger the release pipeline, get the current version from the `databricks_poc_template/__init__.py` file and tag the current code version:
```
git tag -a v<your-project-version> -m "Release tag for version <your-project-version>"
git push origin --tags
```
