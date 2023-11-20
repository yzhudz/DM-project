# Synthetic Dataset Generator

# Overview
This Python script generates a synthetic dataset using a combination of Zipf and exponential distributions. The generated dataset can be used for testing, simulations, or as a placeholder for real data in various data processing or data analysis tasks.

# Features
* Generates keys based on a Zipf distribution.
* Associates each key with a set of values derived from an exponential distribution.
* Supports data skew by modifying the distribution of values.
* Writes the generated dataset to a file in a customizable format.
* Handles large datasets by writing data in chunks.

# Requirements
* Python 3.x
* NumPy library

# Usage
To use this script, follow these steps:

1. **Install NumPy**: If you don't have NumPy installed, you can install it using pip:

```commandline
pip install numpy
```

2. **Configure Parameters**: Before running the script, you may configure the following parameters in the main() function:

* num_entries: Total number of entries in the dataset.
* num_values: Number of values associated with each key.
* zipf_param: Exponent parameter 's' for the Zipf distribution.
* exp_scale: Scale parameter for the exponential distribution.
* file_name: The name of the output file.
* chunk_size: Number of entries per chunk. Adjust based on memory constraints.
* skew: Boolean flag to apply data skew.

3. **Run the Script**: Execute the script in your Python environment:

# File Format
The generated file follows this format:

* Each line represents an entry.
* The first value in each line is the key (from the Zipf distribution).
* Following the key are the associated values (from the exponential distribution), potentially skewed if enabled.

# Example
An example line in the generated file might look like this if skew is enabled:

```
123 450 1500 3200 5000 9800
```
Where 123 is the key, and the subsequent numbers are the skewed values.
