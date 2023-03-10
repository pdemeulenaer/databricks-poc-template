install:
	python -m pip install --upgrade pip &&\
        pip install -e .[local,test]
        
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
	dbx deploy template-etl-workflow
	dbx launch template-etl-workflow --trace	

train_workflow_dev:
	dbx deploy template-train-workflow-dev
	dbx launch template-train-workflow-dev --trace	

train_workflow_staging:
	dbx deploy template-train-workflow-staging
	dbx launch template-train-workflow-staging --trace			

inference_dev:
	dbx deploy template-inference-workflow-dev
	dbx launch template-inference-workflow-dev --trace	

inference_uat: # TODO:
	dbx deploy template-inference-workflow-uat
	dbx launch template-inference-workflow-uat --trace	

inference_prod: # TODO:
	dbx deploy template-inference-workflow-prod
	dbx launch template-inference-workflow-prod --trace	

transition_prod:
	dbx deploy template-transition-to-prod-workflow
	dbx launch template-transition-to-prod-workflow --trace	

monitoring_dev:
	dbx deploy template-monitoring-workflow-dev
	dbx launch template-monitoring-workflow-dev --trace		

monitoring_uat: # TODO:
	dbx deploy template-monitoring-workflow-uat
	dbx launch template-monitoring-workflow-uat --trace	

monitoring_prod: # TODO:
	dbx deploy template-monitoring-workflow-prod
	dbx launch template-monitoring-workflow-prod --trace					

message:
	echo hello $(foo)

all: install lint test