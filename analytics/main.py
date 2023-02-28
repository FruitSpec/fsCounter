from analytics.analyzer import *
from analytics.tools.realtime_pipline import run_real_time
import time


def main_config(
        configs_folder="/home/fruitspec-lab/FruitSpec/Code/fsCounter/vision/pipelines/config/size_comparison_pipe",
        skip_both=False, analyze=True, finished=[], analyze_only=False):
    """
    this function is for running multiple configurations in order to compare them
    :param configs_folder: folder with different configuration files
    :param skip_both: flag if cfg in finished will skip analyzer as well
    :param analyze: flag for running alanyzer
    :param finished: files to pass
    :param analyze_only: flag for running only the analysis
    :return:
    """
    args = OmegaConf.load(os.getcwd() + '/config/runtime.yml')
    for cfg in os.listdir(configs_folder):
        cfg_path = os.path.join(configs_folder, cfg)
        if not cfg in finished and not analyze_only:
            s_t = time.time()
            run_real_time(3, cfg_path)
            print("total time: ", time.time() - s_t)
        elif skip_both:
            continue
        if not analyze:
            continue
        suffix = f'{cfg[cfg.index("_", cfg.index("_") + 1):].split(".")[0]}.csv'
        measures_name = f'measures{suffix}'
        analysis = [phenotyping_analyzer('side1', measures_name),
                    phenotyping_analyzer('side2', measures_name),
                    commercial_analyzer('side1', measures_name),
                    commercial_analyzer('side2', measures_name)]
        df = pd.DataFrame()
        for obj in analysis:
            if obj.validation() == False:
                continue
            obj.run()
            df = pd.concat([df, obj.get_results()], axis=0)
        df.to_csv(os.path.join(args.output_path, f'results{suffix}'), index=False)
    print("finito")


def main():
    args = OmegaConf.load(os.getcwd() + '/config/runtime.yml')
    run_real_time()
    analysis = [phenotyping_analyzer('side1'),
                phenotyping_analyzer('side2'),
                commercial_analyzer('side1'),
                commercial_analyzer('side2')]
    df = pd.DataFrame()
    for obj in analysis:
        if obj.validation() == False:
            continue
        obj.run()
        df = pd.concat([df, obj.get_results()], axis=0)
    df.to_csv(os.path.join(args.output_path, 'results.csv'), index=False)
    print("finito")


if __name__ == "__main__":
    # main_config(skip_both=False, analyze=True, finished=[], analyze_only=True)
    main_config("/home/fruitspec-lab/FruitSpec/Code/fsCounter/vision/pipelines/config/report",
                finished=[],
                analyze=False, skip_both=True)
