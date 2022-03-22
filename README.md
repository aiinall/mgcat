# MGCAT

MGCAT for predicting potential lncRNA-miRNA interactions.

## Overview

This repository is organised as follows:

+ `data/` contains two benchmark datasets;
    + `DB1/`
    + `DB2/`
+ `code/` contains the code needed to run MGCAT;
    + `requirements.txt` contains the environmental dependencies for this implementation.

## Requirements

The implementation is tested under Python 3.8, with the following packages installed:

* python==3.8.12

* pytorch==1.10.2

* torch_geometric==2.0.3

* pandas==1.3.5

* numpy== 1.22.2

* scikit-learn==1.0.2

To get the environment settled quickly, run:

    pip install -r requirements.txt

## Usage

Download code and data, then execute the following command:

    cd code/
    python main.py

