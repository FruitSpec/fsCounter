import argparse

def make_parser():
    parser = argparse.ArgumentParser("Runner")
    parser.add_argument("-m", "--movie_path", type=str, default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("-o", "--output_folder", type=str, default=None)

    parser.add_argument("-d", "--data_dir", type=str, default=None)
    parser.add_argument("-gt", "--coco_gt", type=str, default=None)
    parser.add_argument("-name", "--ds_name", type=str, default="val")
    parser.add_argument("-eb", "--eval_batch", type=int, default=8)


    return parser