import pandas as pd
from abc import abstractmethod

import numpy as np
from omegaconf import OmegaConf
# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
from analytics.tools.utils import *


class Analyzer():
    def __init__(self):
        self.map = OmegaConf.load(os.getcwd() + '/tools/mapping.yml')
        args = OmegaConf.load(os.getcwd() + '/config/runtime.yml')
        self.scan_pre = os.path.join(args.analysis_path, 'pre')
        self.scan_post = os.path.join(args.analysis_path, 'post')
        self.fruit_type = args.fruit_type

        self.side = None
        self.results = pd.DataFrame()

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def validation(self):
        return True

    @staticmethod
    def diff_size(nonPicked_ratio, all_measures, nonPicked_measures):
        """
        :param nonPicked_ratio: number of fruit ratio between post-number and pre-number
        :param all_measures: size measures of pre scan
        :param nonPicked_measures: size measures of post scan
        :return: miu and sigma of picked size (+weight) measures by 3 different calculations
        """

        def diff_size_by_normal():
            picked_ratio = 1 - nonPicked_ratio
            miu_all = np.nanmean(all_measures)
            sigma_all = np.nanstd(all_measures)
            miu_nonPicked = np.nanmean(nonPicked_measures)
            sigma_nonPicked = np.nanstd(nonPicked_measures)
            miu_picked = (1 / picked_ratio) * miu_all - (nonPicked_ratio / picked_ratio) * miu_nonPicked
            sigma_picked = np.sqrt(
                (1 / picked_ratio) ** 2 * sigma_all ** 2 + (nonPicked_ratio / picked_ratio) ** 2 * sigma_nonPicked ** 2)
            return miu_picked, sigma_picked

        # [1] By normal assumption
        miu_picked, sigma_picked = diff_size_by_normal()
        all_measures_weight = predict_weight_values(0, 0, all_measures)
        all_measures = all_measures_weight
        nonPicked_measures_weight = predict_weight_values(0, 0, nonPicked_measures)
        nonPicked_measures = nonPicked_measures_weight
        weight_miu, weight_sigma = diff_size_by_normal()

        # [2] By kde, not robust enough
        # x_values = np.linspace(all_measures.min(), all_measures.max(), num=int(all_measures.max() - all_measures.min()))
        # kde_all = gaussian_kde(all_measures)(x_values)
        # kde_nonPicked = gaussian_kde(nonPicked_measures)(x_values)
        # kde_picked = (1 / picked_ratio) * kde_all - (nonPicked_ratio / picked_ratio) * kde_nonPicked
        # kde_miu = np.sum(kde_picked * x_values)
        # kde_sigma = np.sqrt(np.sum(kde_picked * (x_values - kde_miu) ** 2))

        # # [3] By hist
        # hist_all, bins = np.histogram(all_measures,normed=True)
        # hist_nonPicked, _ = np.histogram(nonPicked_measures, normed=True, bins=bins)
        # hist_picked = (1 / picked_ratio) * hist_all - (nonPicked_ratio / picked_ratio) * hist_nonPicked
        # hist_picked = hist_picked / np.sum(hist_picked)
        # bins = [(var + bins[i + 1]) / 2 for i, var in enumerate(bins) if i + 1 != len(bins)]
        # hist_miu = np.sum(hist_picked * bins)
        # hist_sigma = np.sqrt(np.sum(hist_picked * (bins - hist_miu) ** 2))

        return (miu_picked, sigma_picked, weight_miu, weight_sigma), (None, None), (None, None)

    @staticmethod
    def diff_color(pre_color, post_color, picked_count):
        """
        :param pre_color: pre scan color values by a set of int values that refers to fruit color ranges
        :param post_color: post scan color values by a set of int values that refers to fruit color ranges
        :param picked_count: int, number of picked fruits in a unit
        :return: number of picked fruits for each bin-color
        """
        hist_all, _ = np.histogram(pre_color, density=False, bins=[1, 2, 3, 4, 5, 6])
        hist_nonPicked, _ = np.histogram(post_color, density=False, bins=[1, 2, 3, 4, 5, 6])
        hist_picked = hist_all - hist_nonPicked
        hist_picked = [max(i, 0) for i in hist_picked]
        hist_picked = hist_picked / np.sum(hist_picked)
        picked_bins = picked_count * hist_picked

        try:
            bin1, bin2, bin3, bin4, bin5 = int(picked_bins[0]), int(picked_bins[1]), int(picked_bins[2]), int(
                picked_bins[3]), \
                picked_count - (int(picked_bins[0]) + int(picked_bins[1]) + int(picked_bins[2]) + int(picked_bins[3]))
        except:
            bin1, bin2, bin3, bin4, bin5 = 0, 0, 0, 0, 0

        return bin1, bin2, bin3, bin4, bin5

    @staticmethod
    def valid_output(arg1, arg2):
        if arg1[0] is None or arg2[0] is None:
            return False
        return True

    def calc_diff_values(self, pre, post, plot_id):
        """
        Handling diff calcs between pre and post
        :param pre: pre [0]-count, [1]-size, [2]-color
        :param post: post [0]-count, [1]-size, [2]-color
        :param plot_id: unit id
        """
        picked_count = pre[0] - post[0]
        # check relevance to calc diff color and size
        if picked_count < 0:
            self.results = append_results(self.results, [plot_id, picked_count] + [None] * 9)
            return
        elif picked_count == 0:
            self.results = append_results(self.results, [plot_id, picked_count] + [0] * 9)
            return
        try:
            (size_value_miu, size_value_sigma, weight_miu, weight_sigma), (kde_miu, kde_sigma), (
                hist_miu, hist_sigma) = Analyzer.diff_size(
                post[0] / pre[0], pd.DataFrame(pre[1]), pd.DataFrame(post[1]))
            bin1, bin2, bin3, bin4, bin5 = Analyzer.diff_color(pre[2], post[2], picked_count)
        except Exception as e:
            self.results = append_results(self.results, [plot_id, picked_count] + [0] * 9)
            print(f"{plot_id} : {repr(e)}")
            return
        self.results = append_results(self.results,
                                      [plot_id, picked_count, size_value_miu, size_value_sigma, weight_miu,
                                       weight_sigma, bin1, bin2, bin3, bin4, bin5])

    def calc_single_values(self, pre, plot_id):
        count = pre[0]
        size_value_miu = np.nanmean(pre[1])
        size_value_sigma = np.nanstd(pre[1])
        weight_miu, weight_sigma = predict_weight_values(size_value_miu, size_value_sigma)
        color_hist, _ = np.histogram(pre[2], normed=False, bins=[1, 2, 3, 4, 5, 6])
        color_hist = [max(i, 0) for i in color_hist]
        color_hist = color_hist / np.sum(color_hist)
        count_color_bins = count * color_hist
        bin1, bin2, bin3, bin4, bin5 = int(count_color_bins[0]), int(count_color_bins[1]), int(
            count_color_bins[2]), int(
            count_color_bins[3]), \
            count - (int(count_color_bins[0]) + int(count_color_bins[1]) + int(count_color_bins[2]) + int(
                count_color_bins[3]))

        self.results = append_results(self.results,
                                      [plot_id, count, size_value_miu, size_value_sigma, weight_miu,
                                       weight_sigma, bin1, bin2, bin3, bin4, bin5])

    def get_results(self):
        return self.results

    def get_pre_post(self, id, pre, post):
        """
        Utility to get the raw values of pre and post
        """
        return {'id': id,
                'pre': {'count': pre[0],
                        'size': pre[1],
                        'color': pre[2]},
                'post': {'count': post[0],
                         'size': post[1],
                         'color': post[2]}}


class phenotyping_analyzer(Analyzer):
    """
    analysis for phenotype needs
    """

    def __init__(self, measures_name="measures.csv"):
        super(phenotyping_analyzer, self).__init__()
        self.indices = self.map[self.fruit_type].syngenta.phenotyping.rows
        self.measures_name = measures_name
        self.tree_plot_map = pd.read_csv(os.getcwd() + self.map.plot_code_map_path)
        self.active_tracks = []
        self.df_debug_plots = pd.DataFrame()

    def validation(self):
        flag = True
        # validate manual slicers
        for scan in [self.scan_pre, self.scan_post]:
            for row in self.indices:
                try:
                    json_path = os.path.join(scan, row,
                                             [i for i in os.listdir(os.path.join(scan, row)) if 'slice_data' in i][0])
                except Exception:
                    print(f'{scan.split("/")[-1]} - {row} - NOT EXIST!')
                    continue

                try:
                    trees = slice_to_trees(json_path, None, None, w=1080, h=1920)[0]
                    exist_plots = len(trees['tree_id'].unique())
                except ValueError as e:
                    print(f'{scan.split("/")[-1]} - {row} - {repr(e)}')
                    continue

                GT_plots = self.map[self.fruit_type].syngenta.phenotyping.plot_per_row[row]
                if GT_plots == exist_plots:
                    print(f'{scan.split("/")[-1]} - {row} - completed!')
                else:
                    print(f'{scan.split("/")[-1]} - {row} - {exist_plots}/{GT_plots} - NOT MATCHED')
                    flag = False
        return flag

    def map_tree_into_plot(self, row, tree, type):
        df = self.tree_plot_map
        t = df[(df['row'] == int(row)) & (df['tree_id'] == tree) & (df['type'] == type)]
        try:
            res = t.iloc[0]['plot']
        except:
            print('ERROR in plot mapping, row: ', row, 'tree_num:', tree)
            return 0000
        return res

    def set_df_debug_plots(self, df):
        self.df_debug_plots = pd.concat([self.df_debug_plots, df], axis=0)

    def set_active_tracks(self, _ids):
        # ids to not use on the next plot process
        self.active_tracks = _ids

    def get_dict_plots(self, path):
        """
        :param path: path to the real-time files
        :return: dictionary of plots with their processed values
        """

        def get_sets(row):
            row_path = os.path.join(path, row)
            if os.path.exists(os.path.join(path, row, self.measures_name)):
                df_res = open_measures(row_path, self.measures_name)
                trees, borders = get_trees(row_path)

                # filter according plot's implications
                if row == '9':
                    df_res = filter_trackers(df_res, dist_threshold=0.9)
                else:
                    df_res = filter_trackers(df_res, dist_threshold=0)

            else:
                print(f"NO MEASURES FILE - {os.path.join(path, row, self.measures_name)} - PLOTS' ROW REMOVED ")
                return None, None, None

            return (df_res, trees, borders)

        dict_plots = {}
        for row in self.indices:
            self.active_tracks = []
            df_res, trees, borders = get_sets(row)
            # In case there is no data for the current row
            if df_res is None:
                continue

            for tree_id, df_tree in trees:
                plot_id = self.map_tree_into_plot(row, tree_id, self.fruit_type)

                df_border = borders[borders.tree_id == tree_id]
                if not len(df_border):
                    df_border = None

                # condition on same track_id in 2 plots
                df_det = df_res[~df_res['track_id'].isin(self.active_tracks)]
                self.current_values = {'row': row, 'plot_id': plot_id, 'scan': path.split('/')[-1]}
                _counter, _size, _color = trackers_into_values(df_det, df_tree, df_border, self)

                if not plot_id in dict_plots:
                    dict_plots[plot_id] = {'count': 0, 'size': [], 'color': []}
                dict_plots[plot_id]['count'] += _counter
                dict_plots[plot_id]['size'] += _size.values.tolist()
                dict_plots[plot_id]['color'] += _color.values.tolist()

        return dict_plots

    def run(self):
        """
        Execute all phenotyping plots according mapping_config, extract its count,size,color values after diff
        Update self.results with all units' results
        """
        df_sum = pd.DataFrame()
        plots_pre = self.get_dict_plots(self.scan_pre)
        plots_post = self.get_dict_plots(self.scan_post)

        for pre_item, post_item in zip(plots_pre.items(), plots_post.items()):
            # pre ,post - [0]-count, [1]-size, [2]-color , [3]- plot id
            pre = (pre_item[1]['count'], pre_item[1]['size'], pre_item[1]['color'], pre_item[0])
            post = (post_item[1]['count'], post_item[1]['size'], post_item[1]['color'], post_item[0])
            if not Analyzer.valid_output(pre, post):
                df_sum = append_results(df_sum, [pre[0]] + [None] * 10)
                continue
            # dict =  self.get_pre_post(pre[3],pre,post)
            self.calc_diff_values(pre, post, pre[3])


class commercial_analyzer(Analyzer):
    """
        analysis for commercial fruits needs
    """

    def __init__(self, customer, measures_name="measures.csv"):
        super(commercial_analyzer, self).__init__()
        self.customer = customer
        self.indices = self.map[self.fruit_type][customer].commercial.rows
        self.measures_name = measures_name

    @staticmethod
    def get_aggregation(path, rows, measures_name):
        counter = 0
        size = pd.DataFrame()
        color = pd.DataFrame()

        for row in rows:
            if not os.path.exists(os.path.join(path, row, measures_name)):
                print(f"NO MEASURES FILE - {os.path.join(path, row, measures_name)} - REMOVED ")
                continue
            df_res = open_measures(os.path.join(path, row), measures_name)
            df_res = filter_trackers(df_res, dist_threshold=0)
            _counter, _size, _color = trackers_into_values(df_res)
            counter += _counter
            size = pd.concat([size, _size], axis=0)
            color = pd.concat([color, _color], axis=0)

        if counter == 0 or size.empty or color.empty:
            raise FileNotFoundError

        return (counter, size.values, color.values)



    def run(self):
        """
        Execute all commercial units according mapping_config, extract its count,size,color values after diff
        Update self.results with all units' results
        """

        def run_pre_post():
            post = commercial_analyzer.get_aggregation(self.scan_post, rows, self.measures_name)
            # One of the file measures does not exist

            self.calc_diff_values(pre, post, key)

        for key, rows in self.indices.items():
            try:
                pre = commercial_analyzer.get_aggregation(self.scan_pre, rows, self.measures_name)
                if self.customer == 'syngenta':
                    run_pre_post()
                else:
                    self.calc_single_values(pre, key)

            # One of the file measures does not exist
            except FileNotFoundError:
                self.results = append_results(self.results, [key] + [None] * 10)
                continue
            # dict =  self.get_pre_post(key,pre,post)
