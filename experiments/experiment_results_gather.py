import os
from typing import List

import pandas as pd


def merge_experiment_results(root_folders: List[str]) -> None:
    """
    Merge all experiment_results.csv files from the specified root folders into a single CSV file
    for each root folder.

    Args:
        root_folders (List[str]): List of root folders containing the experiment fold subfolders.

    Returns:
        None
    """
    for root_folder in root_folders:
        all_data = []

        for fold_name in os.listdir(root_folder):
            fold_path = os.path.join(root_folder, fold_name)
            if os.path.isdir(fold_path) and fold_name.startswith('fold_'):
                csv_path = os.path.join(fold_path, 'experiment_results.csv')
                if os.path.isfile(csv_path):
                    df = pd.read_csv(csv_path)
                    all_data.append(df)

        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)
            merged_df = merged_df.sort_values(by='Fold')
            output_path = os.path.join(root_folder, 'merged_experiment_results.csv')
            merged_df.to_csv(output_path, index=False)
            print(f"Merged CSV saved to {output_path}")


def main() -> None:
    """
    Main function to define the list of root folders and initiate the merging process.

    Returns:
        None
    """
    root_folders = [
        "<experiment_folders_paths>",
    ]

    merge_experiment_results(root_folders)


if __name__ == "__main__":
    main()
