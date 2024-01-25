import os
import pandas as pd
from datetime import datetime
from vision.tools.utils_general import find_subdirs_with_string
from vision.misc.help_func import validate_output_path
import re

def create_dataframe_from_paths(paths):
    """
    Create a dataframe from a list of paths.
    The 'row' column is extracted from the last directory in each path,
    and the 'column' is extracted from the third-to-last directory in each path.

    Parameters:
    paths (list): List of file paths

    Returns:
    pd.DataFrame: DataFrame with specified columns
    """
    # Initialize lists to hold the extracted data
    ids, blocks, rows, trees, sides, tree_ids, Fs = [], [], [], [], [], [], []

    # Iterate over each path
    for path in paths:
        # Split the path into parts
        path_parts = path.split(os.sep)

        # Extract relevant parts of the path
        row = path_parts[-1]  # Last directory
        block = path_parts[-3]  # Third-to-last directory

        # Append extracted parts to lists
        ids.append(None)  # Placeholder for 'id'
        blocks.append(block)
        rows.append(row)
        trees.append(None)  # Placeholder for 'tree'
        sides.append(None)  # Placeholder for 'side'
        tree_ids.append(None)  # Placeholder for 'tree_id'
        Fs.append(None)  # Placeholder for 'F'

    # Create a DataFrame
    df = pd.DataFrame({
        'id': ids,
        'block': blocks,
        'row': rows,
        'tree': trees,
        'side': sides,
        'tree_id': tree_ids,
        'F': Fs
    })

    df['row_int'] = df['row'].apply(lambda s: int(re.search(r'\d+', s).group()))
    df = df.sort_values(by=['block', 'row_int'])

    return df

def generate_data_meta_template(blocks_path, trees_in_row = 1, save = False):

    search_string = 'row_'
    paths = find_subdirs_with_string(blocks_path, search_string)
    df = create_dataframe_from_paths(paths)


    # Duplicating each row 'trees_in_row' times
    df_duplicated = pd.DataFrame()

    for _, row in df.iterrows():
        duplicated_rows = pd.concat([pd.DataFrame([row])] * trees_in_row, ignore_index=True)
        # Update the 'tree_id' for each set of duplicated rows
        duplicated_rows['tree_id'] = range(1, trees_in_row + 1)
        df_duplicated = df_duplicated.append(duplicated_rows, ignore_index=True)

    # Update the 'id' column with a serial number
    df_duplicated['id'] = range(1, len(df_duplicated) + 1)

    if save:
        output_path = os.path.join(blocks_path, 'Data_files', f'data_meta_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
        validate_output_path(output_path)
        df_duplicated.to_csv(output_path, index=False)
        print (f'Saved {output_path}')
    return df_duplicated


if __name__ == "__main__":
    blocks_path = r'/home/fruitspec-lab-3/FruitSpec/Data/ground_oranges/Israel'
    df = generate_data_meta_template(blocks_path, trees_in_row = 2, save = True)