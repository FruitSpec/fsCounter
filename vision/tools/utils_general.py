import os


def variable_exists(var_name):
    '''
    Check if a variable exists.
    '''
    try:
        exec(f'{var_name}')
    except NameError:
        return False
    else:
        return True
    
def find_subdirs_with_file(folder_path, file_name, return_dirs=True, single_file=True):
    """
    Find file in a folder.
    Return parent dir of the file if 'return_dirs' == True, else return the file path.
    Input file name can be a file name as 'file_name.csv' or just suffix as '.csv'.
    If a single file is expected (single_file=True) and more than one found, raise error
    """
    subdirs_with_file = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file_name in file:
                if return_dirs:
                    subdir_path = os.path.abspath(root)
                    subdirs_with_file.append(subdir_path)
                else:
                    file_path = os.path.join(root, file)
                    subdirs_with_file.append(file_path)

    if len(subdirs_with_file) == 0:
        raise FileNotFoundError(f"No matching files were found in the folder '{folder_path}'.")

    if len(subdirs_with_file) > 1:
        if single_file:
            raise ValueError(f"Multiple files found\n{subdirs_with_file}. Expected a single file.")
        else: return subdirs_with_file

    else: return subdirs_with_file[0] # if there is only one file

