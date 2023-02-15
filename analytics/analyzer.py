from abc import abstractmethod

import sys

sys.path.append('/home/yotam/FruitSpec/Code/Dana/fsCounter/vision')
sys.path.append('/home/yotam/FruitSpec/Code/Dana/fsCounter/analytics')

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from analytics.tools.utils import *


class Analyzer():
    def __init__(self):
        self.map = OmegaConf.load(os.getcwd() + '/tools/mapping_config.yml')
        args = OmegaConf.load(os.getcwd() + '/config/runtime.yml')
        self.scan_pre = args.pre_path
        self.scan_post = args.post_path
        self.output_path = args.output_path
        self.fruit_type = 'tomato'

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def diff_size(nonPicked_ratio, all_measures, nonPicked_measures):
        picked_ratio = 1 - nonPicked_ratio
        if picked_ratio == 0:
            return (0, 0), (0, 0), (0, 0)

        # By miu,sigma , assume normalization
        miu_all = all_measures.mean()
        sigma_all = all_measures.std()
        miu_nonPicked = nonPicked_measures.mean()
        sigma_nonPicked = nonPicked_measures.std()
        miu_picked = (1 / picked_ratio) * miu_all - (nonPicked_ratio / picked_ratio) * miu_nonPicked
        sigma_picked = np.sqrt((1 / picked_ratio) ** 2 * sigma_all ** 2 - (nonPicked_ratio / picked_ratio) ** 2 * sigma_nonPicked ** 2)

        # By kde
        # x_values = np.linspace(all_measures.min(), all_measures.max(), num=int(all_measures.max() - all_measures.min()))
        # kde_all = gaussian_kde(all_measures)(x_values)
        # kde_nonPicked = gaussian_kde(nonPicked_measures)(x_values)
        # kde_picked = (1 / picked_ratio) * kde_all - (nonPicked_ratio / picked_ratio) * kde_nonPicked
        # kde_miu = np.sum(kde_picked * x_values)
        # kde_sigma = np.sqrt(np.sum(kde_picked * (x_values - kde_miu) ** 2))

        # By hist
        hist_all, bins, p = plt.hist(all_measures, density=True)
        hist_nonPicked = plt.hist(nonPicked_measures, density=True, bins=bins)[0]
        hist_picked = (1 / picked_ratio) * hist_all - (nonPicked_ratio / picked_ratio) * hist_nonPicked
        hist_picked = hist_picked / np.sum(hist_picked)
        bins = [(var + bins[i + 1]) / 2 for i, var in enumerate(bins) if i + 1 != len(bins)]
        hist_miu = np.sum(hist_picked * bins)
        hist_sigma = np.sqrt(np.sum(hist_picked * (bins - hist_miu) ** 2))

        plt.close()
        return (miu_picked, sigma_picked), (None, None), (hist_miu, hist_sigma)

    @staticmethod
    def diff_color(pre_color, post_color, picked_count):
        hist_all = plt.hist(pre_color, density=False, bins=[1, 2, 3, 4, 5])[0]
        hist_nonPicked = plt.hist(post_color, density=False, bins=[1, 2, 3, 4, 5])[0]
        plt.close()
        hist_picked = hist_all - hist_nonPicked
        hist_picked = [max(i, 0) for i in hist_picked]
        hist_picked = hist_picked / np.sum(hist_picked)
        picked_bins = picked_count * hist_picked

        try:
            bin1, bin2, bin3, bin4 = int(picked_bins[0]), int(picked_bins[1]), int(picked_bins[2]), picked_count - (int(picked_bins[0]) + int(picked_bins[1]) + int(picked_bins[2]))
        except:
            bin1, bin2, bin3, bin4 = 0, 0, 0, 0

        return bin1, bin2, bin3, bin4

    def map_tree_into_plot(self, row, tree, type):
        df = pd.read_csv(self.map.tomato.phenotyping.plot_code_map_path)
        t = df[(df['row'] == int(row)) & (df['tree_id'] == tree) & (df['type'] == type)]
        try:
            res = t.iloc[0]['plot']
        except:
            print('ERROR in plot mapping, row: ', row, 'tree_num:', tree)
            return 0000
        return res

    def val_manual_slice(self):
        for scan in [self.scan_pre, self.scan_post]:
            for row in os.listdir(scan):
                try:
                    json_path = os.path.join(scan, row, [i for i in os.listdir(os.path.join(scan, row)) if 'slice_data' in i][0])
                except Exception as e:
                    continue
                exist_plots = len(slice_to_trees(json_path, None, None)['tree_id'].unique())
                GT_plots = self.map.tomato.phenotyping.plot_per_row[row]
                if GT_plots == exist_plots:
                    print(f'{scan} - {row} - completed!')
                else:
                    print(f'{scan} - {row} - {exist_plots}/{GT_plots}')

    def iter_plots(self, path, iter_side):
        for row in iter_side:
            row_path = os.path.join(path, row)
            df_res = open_measures(row_path)
            trees = get_trees(row_path)
            for tree_id, df_tree in trees:
                counter, size, color = trackers_into_values(df_res, df_tree)
                plot_id = Analyzer.map_tree_into_plot(row, tree_id, self.fruit_type)
                yield (plot_id, counter, size, color)


class commercial_analyzer(Analyzer):

    def __init__(self, side):
        super(commercial_analyzer, self).__init__()
        if side == 'side1':
            self.side = 'side1'
            self.indices = self.map.tomato.commercial.side1
        elif side == 'side2':
            self.side = 'side2'
            self.indices = self.map.tomato.commercial.side2

    def run(self):
        df_sum = pd.DataFrame()
        for pre, post in zip(self.iter_plots(self.scan_pre, self.indices), self.iter_plots(self.scan_post, self.indices)):
            # pre ,post - [0]-id, [1]-count, [2]-size, [3]-color
            (size_value_miu, size_value_sigma), (kde_miu, kde_sigma), (hist_miu, hist_sigma) = Analyzer.diff_size(post[1] / pre[1], pre[2], post[2])
            picked_count = pre[1] - post[1]
            weight_miu, weight_sigma = predict_weight_values(size_value_miu, size_value_sigma)

            bin1, bin2, bin3, bin4 = Analyzer.diff_color(pre[3], post[3], picked_count)

            df_sum = df_sum.append({"side": self.side,
                                    "plot_id": pre[0],
                                    "count": picked_count,
                                    "avg_size": size_value_miu,
                                    "std_size": size_value_sigma,
                                    "avg_weight": weight_miu,
                                    "std_weight": weight_sigma,
                                    "bin1": bin1,
                                    "bin2": bin2,
                                    "bin3": bin3,
                                    "bin4": bin4}, ignore_index=True)
        return df_sum


class phenotyping_analyzer(Analyzer):
    def __init__(self, side):
        super(phenotyping_analyzer, self).__init__()
        if side == 'side1':
            self.side = 'side1'
            self.indices = self.map.tomato.phenotyping.side1
        elif side == 'side2':
            self.side = 'side2'
            self.indices = self.map.tomato.phenotyping.side2

    def run(self):
        df_sum = pd.DataFrame()
        for key, (row1, row2) in self.indices.items():
            df_res = open_measures(os.path.join(self.scan_pre, row1))
            counter1, size1, color1 = trackers_into_values(df_res)
            df_res = open_measures(os.path.join(self.scan_pre, row2))
            counter2, size2, color2 = trackers_into_values(df_res)
            pre = ((counter1 + counter2), pd.concat([size1, size2], axis=0), pd.concat([color1, color2], axis=0))

            df_res = open_measures(os.path.join(self.scan_post, row1))
            counter1, size1, color1 = trackers_into_values(df_res)
            df_res = open_measures(os.path.join(self.scan_post, row2))
            counter2, size2, color2 = trackers_into_values(df_res)
            post = ((counter1 + counter2), pd.concat([size1, size2], axis=0), pd.concat([color1, color2], axis=0))

            # pre ,post - [0]-count, [1]-size, [2]-color
            (size_value_miu, size_value_sigma), (kde_miu, kde_sigma), (hist_miu, hist_sigma) = Analyzer.diff_size(post[0] / pre[0], pre[1], post[1])
            picked_count = pre[0] - post[0]
            weight_miu, weight_sigma = predict_weight_values(size_value_miu, size_value_sigma)

            bin1, bin2, bin3, bin4 = Analyzer.diff_color(pre[2], post[2], picked_count)

            df_sum = df_sum.append({"side": self.side,
                                    "plot_id": key,
                                    "count": picked_count,
                                    "avg_size": size_value_miu,
                                    "std_size": size_value_sigma,
                                    "avg_weight": weight_miu,
                                    "std_weight": weight_sigma,
                                    "bin1": bin1,
                                    "bin2": bin2,
                                    "bin3": bin3,
                                    "bin4": bin4}, ignore_index=True)
        return df_sum


if __name__ == "__main__":
    # run_real_time()
    analyzer = Analyzer()
    analyzer.val_manual_slice()

    phen_1 = phenotyping_analyzer('side1').run()
    phen_2 = phenotyping_analyzer('side2').run()
    com_1 = commercial_analyzer('side1').run()
    com_2 = commercial_analyzer('side2').run()
    df = pd.concat([phen_1, phen_2, com_1, com_2], axis=0)
    df.to_csv(os.path.join(analyzer.output_path, '.csv'))
