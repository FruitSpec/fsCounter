import pandas as pd
import os
from vision.misc.help_func import read_json

DIR = '/home/lihi/FruitSpec/Data/customers/SUMGLD/R2A/'

path_jai_all_slices = os.path.join(DIR, 'all_slices.csv')
path_frames_alignment_jai_zed = os.path.join(DIR, 'jai_zed.json')
path_translation = os.path.join(DIR, 'alignment.csv')

all_slices = pd.read_csv(path_jai_all_slices, index_col = 0)
df_translation = pd.read_csv(path_translation, index_col = 'frame')
frames_alignment_jai_zed = read_json(path_frames_alignment_jai_zed)

scale = 0.628489583333333
jai_width = 1536
zed_width = 1080
# convert frame id from jai to zed camera:
for index, row in all_slices.iterrows():

    jai_frame_id = row['frame_id']
    zed_frame_id = frames_alignment_jai_zed.get(str(jai_frame_id))
    all_slices.iloc[index]['frame_id'] = zed_frame_id

    translation_x = df_translation.loc[jai_frame_id]['tx']
    start = (row['start'] + translation_x) / (scale)
    #start = int(start * zed_width) # TODO ???
    print(f"JaiStart {row['start']} ZedStart {start}")

print ('Done')




