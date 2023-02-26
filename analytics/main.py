from analytics.analyzer import *
from analytics.tools.realtime_pipline import run_real_time
import time

def main():
    finished = ['pipeline_config_pix_size_median_depth.yaml', 'pipeline_config_pix_size_median_hue.yaml',
                'pipeline_config_pix_size_median_hue_depth.yaml',
 'pipeline_config_reg_5.yaml','pipeline_config_reg_5_depth.yaml', 'pipeline_config_reg_2_hue.yaml',
 'pipeline_config_reg_8_hue.yaml','pipeline_config_reg_5_hue.yaml', 'pipeline_config_pix_size_mean.yaml',
 'pipeline_config_reg_5_hue_depth.yaml','pipeline_config_reg_8_hue_depth.yaml', 'pipeline_config_reg_2_hue_depth.yaml',
 'pipeline_config_reg_2_depth.yaml', 'pipeline_config_reg_2.yaml',
 'pipeline_config_reg_8.yaml','pipeline_config_pix_size_median.yaml', 'pipeline_config_reg_8_depth.yaml']

    skip_both = False
    analyze = True
    args = OmegaConf.load(os.getcwd() + '/config/runtime.yml')
    configs_folder = "/home/fruitspec-lab/FruitSpec/Code/fsCounter/vision/pipelines/config/size_comparison_pipe"
    for cfg in os.listdir(configs_folder):
        cfg_path = os.path.join(configs_folder, cfg)
        if not cfg in finished:
            s_t = time.time()
            run_real_time(3, cfg_path)
            print("total time: ", time.time() - s_t)
        elif skip_both:
            continue
        if not analyze:
            continue
        suffix = f'{cfg[cfg.index("_", cfg.index("_") +1):].split(".")[0]}.csv'
        measures_name = f'measures{suffix}'
        analysis = [#phenotyping_analyzer('side1', measures_name),
                    #phenotyping_analyzer('side2', measures_name),
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

if __name__ == "__main__":
    main()