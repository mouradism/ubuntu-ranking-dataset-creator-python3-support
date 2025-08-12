#!/usr/bin/env bash

# download punkt first
python src/download_punkt.py

python src/create_ubuntu_dataset.py -o train.csv train
python src/create_ubuntu_dataset.py -o test.csv test
python src/create_ubuntu_dataset.py -o valid.csv valid

