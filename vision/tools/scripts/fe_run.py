import os
import pandas as pd
from vision.pipelines.fe_pipeline import run_on_folder
from vision.misc.help_func import validate_output_path


over_write = True
njobs = 5
suffix = ""
print_fids = False
run_only_done_adt = True
min_slice_len = 5

plots_dir = "/media/matans/My Book/FruitSpec/Customers_data/Fowler/daily"
output_path = "/media/matans/My Book/FruitSpec/Customers_data/Fowler/features"
validate_output_path(output_path)
plots = os.listdir(plots_dir)
#rows = ["/home/matans/Documents/fruitspec/sandbox/NWFM/val"]
plots = ['ALLEN000', 'MAZMANIA', 'FREDIANI', 'BLAYNEY0']
for plot in plots:
    plot_folder = os.path.join(plots_dir, plot)
    if os.path.isdir(plot_folder):
        cur_output = os.path.join(output_path, plot)
        validate_output_path(cur_output)
        dates = os.listdir(plot_folder)
        for date in dates:
            date_folder = os.path.join(plot_folder, date)
            if os.path.isdir(date_folder):
                cur_output = os.path.join(output_path, plot, date)
                validate_output_path(cur_output)
                rows = os.listdir(date_folder)
                for row in rows:
                    row_folder = os.path.join(date_folder, row, '1')
                    if os.path.isdir(row_folder):
                        cur_output = os.path.join(output_path, plot, date, row)
                        validate_output_path(cur_output)

                        try:

                            if not os.path.exists(os.path.join(cur_output, 'features.csv')):
                                results = run_on_folder(row_folder, over_write, njobs, suffix, print_fids,
                                                        run_only_done_adt)
                                pd.DataFrame(results).to_csv(os.path.join(cur_output, 'features.csv'))
                                print('Done feature extraction')
                        except:
                            print(f"faild FE on {row_folder}")


