import numpy as np


def generate_and_write_chunk(file, num_entries, num_values, zipf_param, exp_scale, skew):
    """
    Generates a chunk of the synthetic dataset and writes it to a file.

    :param file: The file to write to.
    :param num_entries: The number of entries in the chunk.
    :param num_values: The number of values for each key.
    :param zipf_param: The exponent parameter 's' for the Zipf distribution.
    :param exp_scale: The scale parameter for the exponential distribution.
    """
    keys = np.random.zipf(zipf_param, num_entries)
    for key in keys:
        values = np.random.exponential(exp_scale, num_values)
        if skew:
            value_ints = np.round([values[i] * 100 ** i for i in range(num_values)]).astype(int)
        else:
            value_ints = np.round(values).astype(int)
        line = f"{key} " + " ".join(map(str, value_ints))
        file.write(line + "\n")


def write_dataset_to_file(num_entries, num_values, zipf_param, exp_scale, file_name, chunk_size, skew):
    """
    Writes the synthetic dataset to a file.

    :param num_entries: The total number of entries in the dataset.
    :param num_values: The number of values associated with each key.
    :param zipf_param: The exponent parameter 's' for the Zipf distribution.
    :param exp_scale: The scale parameter for the exponential distribution.
    :param file_name: The name of the file to write to.
    :param chunk_size: The number of entries per chunk.
    """
    with open(file_name, 'w') as f:
        for _ in range(num_entries // chunk_size):
            generate_and_write_chunk(f, chunk_size, num_values, zipf_param, exp_scale, skew)
        # Handle any remaining entries
        remainder = num_entries % chunk_size
        if remainder > 0:
            generate_and_write_chunk(f, remainder, num_values, zipf_param, exp_scale, skew)


def main():
    num_entries = int(1e6)
    num_values = 5
    zipf_param = 1.5
    exp_scale = 5.0
    file_name = "synthetic_dataset.txt"
    chunk_size = 1000000  # Adjust as needed
    skew = 0  # data skew

    write_dataset_to_file(num_entries, num_values, zipf_param, exp_scale, file_name, chunk_size, skew)
    print(f"Synthetic dataset written to {file_name}")


if __name__ == "__main__":
    main()
