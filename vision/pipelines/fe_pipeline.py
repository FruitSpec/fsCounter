import os

import pandas as pd

from vision.misc.help_func import load_json
from vision.feature_extractor.feature_extractor import create_row_features_fe_pipe
from vision.pipelines.ops.simulator import write_metadata, init_cams, update_arg_with_metadata ,get_metadata_path


def run(args):
    args, metadata = update_arg_with_metadata(args)
    zed_cam, rgb_jai_cam, jai_cam = init_cams(args)

    #df = create_row_features_fe_pipe(args.output_folder, zed_shift=args.zed_shift, max_x=600, max_y=900,
    #                                 save_csv=True, block_name=args.block_name, max_z=args.max_z,
    #                                 cameras={"zed_cam": zed_cam, "rgb_jai_cam": rgb_jai_cam, "jai_cam": jai_cam},
    #                                 debug=args.debug.features)
    df = pd.DataFrame()

    zed_cam.close()
    jai_cam.close()
    rgb_jai_cam.close()

    metadata["tree_features"] = False
    write_metadata(args, metadata)
    if "cv" in df.columns:
        only_cv_df = df[["cv", "name"]]
        tree_ids = only_cv_df.loc[:, "name"].copy().apply(lambda x: x.split("_")[1][1:]).values
        only_cv_df["tree_id"] = tree_ids
        only_cv_df = only_cv_df[["name", "tree_id", "cv"]]
        only_cv_df.to_csv(os.path.join(args.output_folder, "trees_cv.csv"))
    else:
        print("problem with ", args.jai.movie_path)

def load_metadata(args):
    meta_data_path = get_metadata_path(args)
    metadata = load_json(meta_data_path)

    return metadata