all: create-venv install-deps

create-venv:
	python3 -m venv venv
	@echo "Virtual environment created!"

jupyter:
	@venv/bin/jupyter lab

install-deps:
	venv/bin/pip install -r requirements.txt

split:
	@venv/bin/python3 src/split_data.py data/data.csv

# Train
train1:
	@venv/bin/python3 -m src.training.train --dataset data/data.csv --config model_config/BreastCancerDiagnosis_config_v100.yaml

train2:
	@venv/bin/python3 -m src.training.train --dataset data/data.csv --config model_config/BreastCancerDiagnosis_config_v110.yaml

train3:
	@venv/bin/python3 -m src.training.train --dataset data/data.csv --config model_config/BreastCancerDiagnosis_config_v120.yaml

# Predict
predict1:
	@venv/bin/python3 src/predict.py --dataset data/data_validation.csv --model models/breast_cancer_diagnosis_v1.0.0.npz --config model_config/BreastCancerDiagnosis_config_v100.yaml

predict2:
	@venv/bin/python3 src/predict.py --dataset data/data_validation.csv --model models/breast_cancer_diagnosis_v1.1.0.npz --config model_config/BreastCancerDiagnosis_config_v110.yaml

predict3:
	@venv/bin/python3 src/predict.py --dataset data/data_validation.csv --model models/breast_cancer_diagnosis_v1.2.0.npz --config model_config/BreastCancerDiagnosis_config_v120.yaml

#accuracy:
#	@venv/bin/python3 src/Bonus/accuracy.py data/houses.csv data/dataset_train.csv

