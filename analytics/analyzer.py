from abc import abstractmethod
from omegaconf import OmegaConf
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
        :return: miu and sigma of picked size measures by 3 different calculations
        """

        picked_ratio = 1 - nonPicked_ratio

        # [1] By miu,sigma , assume normalization
        miu_all = all_measures.mean()
        sigma_all = all_measures.std()
        miu_nonPicked = nonPicked_measures.mean()
        sigma_nonPicked = nonPicked_measures.std()
        miu_picked = (1 / picked_ratio) * miu_all - (nonPicked_ratio / picked_ratio) * miu_nonPicked
        # TODO case when value in sqrt equal to 0
        sigma_picked = np.sqrt((1 / picked_ratio) ** 2 * sigma_all ** 2 - (nonPicked_ratio / picked_ratio) ** 2 * sigma_nonPicked ** 2)

        # [2] By kde, not robust enough
        # x_values = np.linspace(all_measures.min(), all_measures.max(), num=int(all_measures.max() - all_measures.min()))
        # kde_all = gaussian_kde(all_measures)(x_values)
        # kde_nonPicked = gaussian_kde(nonPicked_measures)(x_values)
        # kde_picked = (1 / picked_ratio) * kde_all - (nonPicked_ratio / picked_ratio) * kde_nonPicked
        # kde_miu = np.sum(kde_picked * x_values)
        # kde_sigma = np.sqrt(np.sum(kde_picked * (x_values - kde_miu) ** 2))

        # [3] By hist
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
        """
        :param pre_color: pre scan color values by a set of int values that refers to fruit color ranges
        :param post_color: post scan color values by a set of int values that refers to fruit color ranges
        :param picked_count: int, number of picked fruits in a unit
        :return: number of picked fruits for each bin-color
        """
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
            self.results = append_results(self.results, [self.side, plot_id, picked_count] + [None] * 8)
            return
        elif picked_count == 0:
            self.results = append_results(self.results, [self.side, plot_id, picked_count] + [0] * 8)
            return
        (size_value_miu, size_value_sigma), (kde_miu, kde_sigma), (hist_miu, hist_sigma) = Analyzer.diff_size(post[0] / pre[0], pre[1], post[1])
        weight_miu, weight_sigma = predict_weight_values(size_value_miu, size_value_sigma)
        bin1, bin2, bin3, bin4 = Analyzer.diff_color(pre[2], post[2], picked_count)
        self.results = append_results(self.results, [self.side, plot_id, picked_count, size_value_miu, size_value_sigma, weight_miu, weight_sigma, bin1, bin2, bin3, bin4])

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

    def __init__(self, side, measures_name="measures.csv"):
        super(phenotyping_analyzer, self).__init__()
        if side == 'side1':
            self.side = 'side1'
            self.indices = self.map[self.fruit_type].phenotyping.side1
        elif side == 'side2':
            self.side = 'side2'
            self.indices = self.map[self.fruit_type].phenotyping.side2
        self.measures_name = measures_name
        self.tree_plot_map = pd.read_csv(os.getcwd() + self.map.plot_code_map_path)

    def validation(self):
        flag = True
        # validate manual slicers
        for scan in [self.scan_pre, self.scan_post]:
            for row in self.indices:
                try:
                    json_path = os.path.join(scan, row, [i for i in os.listdir(os.path.join(scan, row)) if 'slice_data' in i][0])
                except Exception:
                    print(f'{scan.split("/")[-1]} - {row} - NOT EXIST!')
                    continue

                try:
                    exist_plots = len(slice_to_trees(json_path, None, None)['tree_id'].unique())
                except ValueError as e:
                    print(f'{scan.split("/")[-1]} - {row} - {repr(e)}')

                GT_plots = self.map[self.fruit_type].phenotyping.plot_per_row[row]
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

    def iter_plots(self, path, iter_side):
        """
        :param path: path to the real-time files
        :param iter_side: list of rows numbers that belong to a row side in greenhouse
        :return: plots iterator with their processed values
        """
        for row in iter_side:
            row_path = os.path.join(path, row)
            try:
                df_res = open_measures(row_path, self.measures_name)
                trees = get_trees(row_path)
            except FileNotFoundError:
                yield None, row
                continue
            for tree_id, df_tree in trees:
                counter, size, color = trackers_into_values(df_res, df_tree)
                plot_id = self.map_tree_into_plot(row, tree_id, self.fruit_type)
                yield (counter, size, color, plot_id)

    def run(self):
        """
        Execute all phenotyping plots according mapping_config, extract its count,size,color values after diff
        Update self.results with all units' results
        """
        df_sum = pd.DataFrame()
        for pre, post in zip(self.iter_plots(self.scan_pre, self.indices), self.iter_plots(self.scan_post, self.indices)):
            # pre ,post - [0]-count, [1]-size, [2]-color , [3]- plot id
            if not Analyzer.valid_output(pre, post):
                df_sum = append_results(df_sum, [self.side, pre[0]] + [None] * 9)
                continue
            # dict =  self.get_pre_post(pre[3],pre,post)
            self.calc_diff_values(pre, post, pre[3])


class commercial_analyzer(Analyzer):
    """
        analysis for commercial fruits needs
    """
    def __init__(self, side, measures_name="measures.csv"):
        super(commercial_analyzer, self).__init__()
        if side == 'side1':
            self.side = 'side1'
            self.indices = self.map[self.fruit_type].commercial.side1
        elif side == 'side2':
            self.side = 'side2'
            self.indices = self.map[self.fruit_type].commercial.side2
        self.measures_name = measures_name

    @staticmethod
    def get_aggreagation(path, rows, measures_name):
        counter = 0
        size = pd.DataFrame()
        color = pd.DataFrame()

        for row in rows:
            df_res = open_measures(os.path.join(path, row), measures_name)
            _counter, _size, _color = trackers_into_values(df_res)
            counter += _counter
            size = pd.concat([size, _size], axis=0)
            color = pd.concat([color, _color], axis=0)

        return (counter, size.values, color.values)

    def run(self):
        """
        Execute all commercial units according mapping_config, extract its count,size,color values after diff
        Update self.results with all units' results
        """
        df_sum = pd.DataFrame()
        for key, rows in self.indices.items():
            try:
                pre = commercial_analyzer.get_aggreagation(self.scan_pre, rows, self.measures_name)
                post = commercial_analyzer.get_aggreagation(self.scan_post, rows, self.measures_name)
            # One of the file measures does not exist
            except FileNotFoundError:
                df_sum = append_results(df_sum, [self.side, key] + [None] * 9)
                continue
            # dict =  self.get_pre_post(key,pre,post)
            self.calc_diff_values(pre, post, key)
