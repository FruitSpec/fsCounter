from vision.tools.utils_general import get_s3_file_paths, download_s3_files, find_subdirs_with_file
from vision.depth.SVO_to_depth import convert_svo_to_depth_bgr_dgr
import os
from vision.tools.utils_general import upload_to_s3
import pandas as pd


# # download svo and annotations files from s3:
s3_path = 's3://fruitspec.dataset/Temp Counter/DEWAGB/'
# output_path = '/home/lihi/FruitSpec/Data/customers/DEWAGD'

# download_s3_files(s3_path, output_path =output_path, suffix='.svo', skip_existing=True)
# download_s3_files(s3_path, output_path =output_path, string_param='slice_data_R', suffix='.json', skip_existing=True)
# download_s3_files(s3_path, output_path =output_path, string_param='all_slices', suffix='.csv', skip_existing=True)
#####################################################################################################################
# local_path = r'/home/lihi/FruitSpec/Data/customers/DEWAGD'
local_path = r'/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD'

paths_svo = find_subdirs_with_file(local_path, file_name ='.svo', return_dirs=False, single_file=False)

# itterate over all svo files and convert to depth:
for svo_path in paths_svo:

    path_output = os.path.join(os.path.dirname(svo_path), "zed_rgb.avi")
    if not os.path.isfile(path_output):
        print(svo_path)
        try:
            local_path_depth, local_path_BGR, local_path_DGR = convert_svo_to_depth_bgr_dgr(svo_path, index=0,rotate=2, save = True)

            # upload to s3:
            output_s3_depth_path = os.path.join(s3_path, os.path.sep.join(local_path_depth.strip('/').split('/')[-4:-1]) )
            res = upload_to_s3(local_path_depth, output_s3_depth_path)
            output_s3_BGR_path = os.path.join(s3_path, os.path.sep.join(local_path_BGR.strip('/').split('/')[-4:-1]) )
            res = upload_to_s3(local_path_BGR, output_s3_BGR_path)
            output_s3_DGR_path = os.path.join(s3_path, os.path.sep.join(local_path_DGR.strip('/').split('/')[-4:-1]) )
            res = upload_to_s3(local_path_DGR, output_s3_DGR_path)

        except Exception as e:
            print(f'Exception: {e}')
            continue

    else:
        print (f'Skip {svo_path}')

print ('Done')
############################




