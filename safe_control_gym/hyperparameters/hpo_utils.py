import os
from datetime import datetime

import pandas as pd


# Function to get the seed folder with the smallest seed number
def get_smallest_seed_folder(algorithm, package, base_dir):
    package_dir = os.path.join(base_dir, algorithm, package)
    seed_folders = [f for f in os.listdir(package_dir) if os.path.isdir(os.path.join(package_dir, f)) and f.startswith('seed')]

    # Extract the seed numbers and sort to find the smallest
    seed_numbers = [(folder, int(folder.split('_')[0][4:])) for folder in seed_folders]
    smallest_seed_folder = min(seed_numbers, key=lambda x: x[1])[0] if seed_numbers else None

    return smallest_seed_folder


def get_smallest_and_latest_seed_folder(output_dir):
    dir = os.path.dirname(output_dir)
    seed_folders = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f)) and f.startswith('seed')]

    if not seed_folders:
        return None  # Return None if no seed folders exist

    # Extract seed numbers and timestamps
    seed_info = []
    for folder in seed_folders:
        parts = folder.split('_')
        seed_number = int(parts[0][4:])  # Extract the seed number after 'seed'
        timestamp = parts[1]  # Extract the timestamp part
        seed_info.append((folder, seed_number, timestamp))

    # Filter only the smallest seed number
    smallest_seed_number = min(seed_info, key=lambda x: x[1])[1]
    smallest_seed_folders = [info for info in seed_info if info[1] == smallest_seed_number]

    # remove current output_dir from the list
    smallest_seed_folders = [info for info in smallest_seed_folders if info[0] != os.path.basename(output_dir)]

    # From the smallest seed folders, find the one with the latest timestamp
    smallest_and_latest_folder = max(
        smallest_seed_folders,
        key=lambda x: datetime.strptime(x[2], '%b-%d-%H-%M-%S')
    )[0]

    # combine the seed folder with the base directory
    smallest_and_latest_folder = os.path.join(dir, smallest_and_latest_folder)

    return smallest_and_latest_folder

# Function to load trials data from the smallest seed folder


def load_trials_data(algorithm, package, base_dir):
    seed_folder = get_smallest_seed_folder(algorithm, package, base_dir)
    if seed_folder:
        file_path = os.path.join(base_dir, algorithm, package, seed_folder, 'hpo', 'trials.csv')
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
    return None
