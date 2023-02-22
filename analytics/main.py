from analytics.analyzer import *
from analytics.tools.realtime_pipline import run_real_time
import time

def main():
    args = OmegaConf.load(os.getcwd() + '/config/runtime.yml')
    s_t = time.time()
    run_real_time()
    print("total time: ", time.time()-s_t)
    measures_name = 'measures_reg_2.csv'
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
    df.to_csv(os.path.join(args.output_path, 'results_pix_med.csv'), index=False)
    print("finito")

if __name__ == "__main__":
    main()