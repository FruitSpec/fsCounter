import os
import boto3

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
    Find file containing a given substring in a folder.
    Return parent dir of the file if 'return_dirs' == True, else return the file path.
    Input substring can be a part of a file name as 'file_name.csv' or just suffix as '.csv'.
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
        raise FileNotFoundError(f"No files containing '{file_name}' were found in the folder '{folder_path}'.")

    if len(subdirs_with_file) > 1:
        if single_file:
            raise ValueError(f"Multiple files found\n{subdirs_with_file}. Expected a single file.")
        else: return subdirs_with_file

    else: return subdirs_with_file[0] # if there is only one file

########   S3 UTILS

def s3_full_path_to_bucket_and_prefix(s3_path):
    path_parts = s3_path.replace('s3://', '').split('/', 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ''
    return bucket_name, prefix


def get_s3_file_paths(s3_path, string_param=None, suffix='.json', include_bucket = False):

    """
    Retrieves S3 file paths that match a specific string parameter and suffix.

    Args:
        s3_path (str): The full S3 path, e.g., 's3://bucket-name/prefix/'.
        string_param (str, optional): The specific string parameter to match in the file paths.
                                     Defaults to None.
        suffix (str, optional): The suffix of the file names to consider. Defaults to '.json'.

    Returns:
        list: A list of S3 file paths that match the specified criteria.
    """
    s3 = boto3.resource('s3')
    bucket_name, prefix = s3_full_path_to_bucket_and_prefix(s3_path)

    files = []

    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=prefix):
        key = obj.key
        if key.endswith(suffix) and (string_param is None or string_param in key):
            if include_bucket:
                files.append(f's3://{bucket_name}/{key}')
            else:
                files.append(key)
            print(key)

    return files


def download_s3_files(s3_path, output_path, string_param=None, suffix='.json', skip_existing=True):
    """
    Downloads S3 files matching the specified criteria and maintains the folder structure locally.

    Args:
        s3_path (str): The full S3 path, e.g., 's3://bucket-name/prefix/'.
        output_path (str): The local output path to save the files.
        string_param (str, optional): The specific string parameter to match in the file paths.
                                     Defaults to None. Example: string_param='slice_data' (files that contain this string)
        suffix (str, optional): The suffix of the file names to consider. Defaults to '.json'.
        skip_existing (bool, optional): Whether to skip downloading a file if it already exists locally.
                                        Defaults to False.
    """
    s3 = boto3.client('s3')
    bucket_name, prefix = s3_full_path_to_bucket_and_prefix(s3_path)
    local_folder = os.path.basename(prefix)
    local_output_path = os.path.join(output_path, local_folder)

    # Create the local output path if it doesn't exist
    os.makedirs(local_output_path, exist_ok=True)

    paginator = s3.get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': bucket_name, 'Prefix': prefix}

    for page in paginator.paginate(**operation_parameters):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith(suffix) and (string_param is None or string_param in key):
                    local_file_path = os.path.join(local_output_path, (key.replace(prefix.lstrip('/'), '')).strip('/'))
                    local_file_dir = os.path.dirname(local_file_path)
                    os.makedirs(local_file_dir, exist_ok=True)  # Create local directories if they don't exist
                    if skip_existing and os.path.exists(local_file_path):
                        print(f"Skipped: {local_file_path} (File already exists)")
                    else:
                        s3.download_file(bucket_name, key, local_file_path)
                        print(f"Downloaded: {local_file_path}")


from botocore.exceptions import ClientError
import logging

import boto3
import os
from botocore.exceptions import ClientError

def upload_to_s3(file_name, full_path_s3_dir):
    """
    Upload a file to an S3 bucket

    :param file_name: File to upload
    :param full_path_s3_dir: Full path to S3 directory
    :return: True if file was uploaded, else False
    """
    # Separate bucket from the rest of the path
    bucket, s3_dir = s3_full_path_to_bucket_and_prefix(full_path_s3_dir)

    # Combine the S3 directory with the file name
    object_name = os.path.join(s3_dir, os.path.basename(file_name))

    # Initialize the S3 client
    s3_client = boto3.client('s3')

    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        print(f"Error uploading file to S3: {e}")
        return False
    print (f"Uploaded: {full_path_s3_dir}")
    return True




if __name__ == "__main__":

    # upload to s3:
    path_s3 = 's3://fruitspec.dataset/Temp Counter/'
    local_path = r'/home/lihi/FruitSpec/code/lihi/fsCounter/vision/lihi_debug_delete_me.py'
    res = upload_to_s3(local_path, path_s3)
    print (res)

######################################
    s3_path = 's3://fruitspec.dataset/Temp Counter/DEWAGB/'
    files_paths = get_s3_file_paths(s3_path, string_param = 'slice_data', suffix='.json', skip_existing=True)

    for file in files_paths:
        print(file)

    output_path = '/home/lihi/FruitSpec/Data/customers/DEWAGD'
    download_s3_files(s3_path, output_path =output_path, string_param='slice_data', suffix='.json', skip_existing=True)



