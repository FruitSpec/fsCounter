from analytics.analyzer import *
# from analytics.tools.realtime_pipline import run_real_time


def accuracy(df, args):
    """
    according gt data calc FS accuracy
    :param df: fs results
    :param args: run directories
    :return:
    """
    gt_df = pd.read_csv(os.path.join(args.output_path, f'{args.scan_date}.csv'))
    acc_df = df.merge(gt_df, how='left', on=['plot_id'])
    acc_df['error_total_w'] = (acc_df['GT-TOTAL-W'] - acc_df['total_weight_kg']) / acc_df['GT-TOTAL-W']
    acc_df['error_count'] = (acc_df['GT-Count'] - acc_df['count']) / acc_df['GT-Count']
    acc_df['error_weight'] = (acc_df['GT-Weight'] - acc_df['weight_avg_gr']) / acc_df['GT-Weight']
    acc_df.to_csv(os.path.join(args.output_path, 'accuracy.csv'), index=False)
    # print(acc_df[['plot_id','error_count']])


def main():
    args = OmegaConf.load(os.getcwd() + '/config/runtime.yml')
    # run_real_time()
    analysis = [phenotyping_analyzer()
                #commercial_analyzer()
                ]
    df = pd.DataFrame()
    for obj in analysis:
        if obj.validation() == False:
            continue
        obj.run()
        df = pd.concat([df, obj.get_results()], axis=0)
    df.to_csv(os.path.join(args.output_path, 'results.csv'), index=False)
    accuracy(df, args)


if __name__ == "__main__":
    main()
