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

train:
	@venv/bin/python3 src/train.py 
#
#accuracy:
#	@venv/bin/python3 src/Bonus/accuracy.py data/houses.csv data/dataset_train.csv

