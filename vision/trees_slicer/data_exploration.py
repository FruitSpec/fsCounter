from vision.tools.utils_general import get_s3_file_paths, download_s3_files, find_subdirs_with_file
import pandas as pd

s3_path = 's3://fruitspec.dataset/Temp Counter/DEWAGB/'
output_path = '/home/lihi/FruitSpec/Data/customers/DEWAGD'

download_s3_files(s3_path, output_path =output_path, suffix='.svo', skip_existing=True)
download_s3_files(s3_path, output_path =output_path, string_param='slice_data_R', suffix='.json', skip_existing=True)
download_s3_files(s3_path, output_path =output_path, string_param='all_slices', suffix='.csv', skip_existing=True)
