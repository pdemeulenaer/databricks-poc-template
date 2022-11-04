install:
	pip install --upgrade pip &&\
		pip install -r unit-requirements.txt &&\
        pip install -e .
        
lint:
	python -m pylint --fail-under=-200.5 --rcfile .pylintrc databricks_poc_template/ tests/ -r n --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" > pylint_report.txt      #pylint --disable=R,C model.py 

format:
	black databricks_poc_template/*.py

test:
	python -m pytest -vv --disable-warnings tests/unit --junitxml=junit/test-results.xml --cov=. --cov-config=.coveragerc --cov-report xml:coverage.xml --cov-report term #--cov-report html:cov_html


# For executions in command line (for test purpose, on interactive clusters)
# train_task: # TODO:
# 	# dbx deploy --jobs=training --deployment-file=./conf/deployment-training.json
# 	# dbx launch --job=training --trace
# 	dbx execute train-workflow --task step-training-task --cluster-name ...	

# validate_task: # TODO:
# 	# dbx deploy --jobs=validation --deployment-file=./conf/deployment-validation.json
# 	# dbx launch --job=validation --trace
# 	dbx execute train-workflow --task step-validation-task --cluster-name ...		

# inference_task: # TODO:
# 	# dbx deploy --jobs=cd-infer-job-staging --deployment-file=./conf/deployment.json
# 	# dbx launch --job=cd-infer-job-staging --trace

# For executions within the CI/CD pipeline
etl_workflow:
	dbx deploy etl-workflow
	dbx launch etl-workflow --trace	

train_workflow_dev:
	dbx deploy train-workflow-dev
	dbx launch train-workflow-dev --trace	

train_workflow_staging:
	dbx deploy train-workflow-staging
	dbx launch train-workflow-staging --trace			

inference_dev: # TODO:
	dbx deploy inference-workflow-dev
	dbx launch inference-workflow-dev --trace	

inference_uat: # TODO:
	dbx deploy inference-workflow-uat
	dbx launch inference-workflow-uat --trace	

inference_prod: # TODO:
	dbx deploy inference-workflow-prod
	dbx launch inference-workflow-prod --trace	

monitoring_dev: # TODO:
	dbx deploy monitoring-workflow-dev
	dbx launch monitoring-workflow-dev --trace		

monitoring_uat: # TODO:
	dbx deploy monitoring-workflow-uat
	dbx launch monitoring-workflow-uat --trace	

monitoring_prod: # TODO:
	dbx deploy monitoring-workflow-prod
	dbx launch monitoring-workflow-prod --trace					

message:
	echo hello $(foo)

all: install lint test