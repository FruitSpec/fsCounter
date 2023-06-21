from vision.tools.utils_general import get_s3_file_paths, download_s3_files, find_subdirs_with_file
import pandas as pd
from vision.depth.SVO_to_depth import convert_svo_to_depth_bgr_dgr


# # download svo and annotations files from s3:
# s3_path = 's3://fruitspec.dataset/Temp Counter/DEWAGB/'
# output_path = '/home/lihi/FruitSpec/Data/customers/DEWAGD'

# download_s3_files(s3_path, output_path =output_path, suffix='.svo', skip_existing=True)
# download_s3_files(s3_path, output_path =output_path, string_param='slice_data_R', suffix='.json', skip_existing=True)
# download_s3_files(s3_path, output_path =output_path, string_param='all_slices', suffix='.csv', skip_existing=True)
#####################################################################################################################
folder_path = r'/home/lihi/FruitSpec/Data/customers/DEWAGD'
paths_svo = find_subdirs_with_file(folder_path, file_name = '.svo', return_dirs=False, single_file=False)

# itterate over all svo files and convert to depth:
for svo_path in paths_svo:
    print(svo_path)
    local_path_depth, local_path_BGR, local_path_DGR = convert_svo_to_depth_bgr_dgr(svo_path, index=0, save = True)

print ('Done')
############################




