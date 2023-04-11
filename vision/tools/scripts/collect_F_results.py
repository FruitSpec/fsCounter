import os
import shutil
import pandas as pd
results_folder = "/media/fruitspec-lab/cam175/results"
out_put_folder = "/media/fruitspec-lab/cam175"

if __name__ == "__main__":
    dfs_list = []
    os.walk(results_folder)
    for root, dirs, files in os.walk(results_folder):
        for file in files:
            if file.endswith(".csv") and "with_F" in file:
                data_file = os.path.join(root, file)
                block_name = os.path.basename(root)
                block_df = pd.read_csv(data_file)
                # block_df["block_name"] = block_name
                full_cv_csv_path = os.path.join(root, f"{file.split('.')[0][:-7]}.csv")
                full_cv_df = pd.read_csv(full_cv_csv_path)
                full_cv_df["row"] = full_cv_df["row_id"].apply(lambda x: x[:-1])
                full_cv_df["F"] = full_cv_df["row"].map(dict(zip(block_df["row_id"].apply(lambda x: x[:-1]),
                                                                 block_df.iloc[:, -1])))
                full_cv_df["block_name"] = block_name
                full_cv_df.drop(["row", "Unnamed: 0"], axis=1, inplace=True)
                dfs_list.append(full_cv_df)
    pd.concat(dfs_list).reset_index().to_csv(os.path.join(out_put_folder, "all_Fs.csv"))