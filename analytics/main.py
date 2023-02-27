from analytics.analyzer import *
from analytics.tools.realtime_pipline import run_real_time

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

if __name__ == "__main__":
    main()