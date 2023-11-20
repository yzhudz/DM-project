# Implementation of Sketch algorithms

# Overview
This project includes Python scripts for implementing and testing three distinct algorithms on different datasets. The aim is to evaluate the performance, accuracy, and efficiency of these algorithms in various scenarios, including handling datasets like Criteo, NBA, and synthetic datasets.

# Project Structure
* coco_imp2.py: Implementation of the COCO algorithm.
* hyperuss.py: Implementation of the HYPERUSS algorithm.
* uss.py: Implementation of the USS algorithm.
* criteo_bucket_num_comparison.py: Script for comparing algorithm performances on the Criteo dataset.
* test_criteo.py: Test script for running algorithms on the Criteo dataset.
* test_nba.py: Test script for running algorithms on an NBA dataset.
* test_syntehtic.py: Test script for running algorithms on synthetic datasets.
* evaluation.py: Contains functions for evaluating algorithm performance.
* ground_truth.py: Script for establishing ground truth for comparison.
* utils.py: Utility functions used across the project.

# Requirements
* Python 3.x
* Additional Python libraries as required (e.g., NumPy, Pandas, mmh3) - please refer to each script for specific import statements.

# Usage
To use these scripts, follow these steps:
1. Install Required Libraries: Ensure all required Python libraries are installed. You can typically install these using pip:
```commandline
pip install numpy pandas mmh3
```
2. Running tests on datasets: Each script can be run individually depending on the requirement. For example, to run a test on the Criteo dataset:
```commandline
python test_criteo.py
```
Please modify the file path of dataset in the test file.