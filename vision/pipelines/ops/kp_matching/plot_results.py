import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(results_path, output_path):
    df = pd.read_csv(results_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns for two side-by-side plots
    sns.lineplot(data=df, x='f_id', y='matches', hue='type', ax=axes[0])
    axes[0].set_title('matches')

    sns.lineplot(data=df, x='f_id', y='tx', hue='type', ax=axes[1])
    axes[1].set_title('tx')

    fig.savefig(os.path.join(output_path, 'res_graph.jpg'), dpi=300)  # Save the figure as a PNG image with 300 DPI


if __name__ == "__main__":
    parent_folder = "/media/matans/My Book/FruitSpec/sandbox/alignment_test"
    folders = os.listdir(parent_folder)
    for folder in folders:
        results_path = os.path.join(parent_folder, folder, 'res.csv')
        output_path = os.path.join(parent_folder, folder)
        plot_results(results_path, output_path)


    #results_path = '/media/matans/My Book/FruitSpec/sandbox/alignment_test/03602060_200823_row_12/res.csv'
    #output_path = '/media/matans/My Book/FruitSpec/sandbox/alignment_test/03602060_200823_row_12'
    #plot_results(results_path, output_path)



