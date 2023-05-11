import pandas as pd
import os
import time

from application.utils.module_wrapper import ModulesEnum, Module, ModuleTransferAction
from application.utils.settings import pipeline_conf, runtime_args, data_conf
from vision.pipelines.adt_pipeline import run
import signal


class AlternativeFlow(Module):

    @staticmethod
    def init_module(sender, receiver, main_pid, module_name):
        super(AlternativeFlow, AlternativeFlow).init_module(sender, receiver, main_pid, module_name)
        signal.signal(signal.SIGTERM, AlternativeFlow.shutdown)
        signal.signal(signal.SIGUSR1, AlternativeFlow.receive_data)
        AlternativeFlow.analyze()

    @staticmethod
    def shutdown(sig, frame):
        pass

    @staticmethod
    def receive_data(sig, frame):
        pass

    @staicmethod
    def analyze():
        # todo check if there is stop signal
        while True:
            found, row = AlternativeFlow.seek_new_row()
            if found:
                row_runtime_args = AlternativeFlow.update_runtime_args(runtime_args, row)
                rc = run(pipeline_conf, row_runtime_args)
                data = {'tracks': rc.tracks, 'alignment': rc.alignment, 'row': row}
                #send results to data manager
                # todo change ModuleTransferAction value
                AlternativeFlow.send_data(ModuleTransferAction.FRUITS_DATA, data)
            else:
                time.sleep(60)


    @staticmethod
    def seek_new_row():
        collected = pd.read_csv(data_conf['collected path'])
        analyzed = pd.read_csv(data_conf['analyzed path'])

        analyzed_list = []
        for k, row in analyzed.iterrows():
            analyzed_list.append(create_str_from_row(row))

        row = None  # in case collected is empty - argument sent in return
        found_new_row = False
        for k, row in collected.iterrows():
            unique_str = create_str_from_row(row)
            if unique_str in analyzed_list:
                continue
            else:
                found_new_row = True
                break

        return found_new_row, row


    @staticmethod
    def update_runtime_args(args, row):

        row_args = args.copy()
        row_folder = os.path.join(data_conf['output path'], row['customer_code'], row['plot_code'],
                                  row['scan_date'], f"row_{row['row']}")
        clip_id = int(row['file_index'])
        row_args.output_folder = row_folder
        row_args.row_name = clip_id
        row_args.zed.movie_path = os.path.join(row_folder, f"ZED_{clip_id}.mkv")
        row_args.depth.movie_path = os.path.join(row_folder, f"Depth_{clip_id}.mkv")
        row_args.jai.movie_path = os.path.join(row_folder, f"Result_FSI_{clip_id}.mkv")
        row_args.rgb_jai.movie_path = os.path.join(row_folder, f"Result_RGB_{clip_id}.mkv")
        row_args.sync_data_log_path = os.path.join(row_folder, f"jaized_timestamps_{clip_id}.log")
        row_args.slice_data_path = os.path.join(row_folder, f"Result_FSI_{clip_id}_slice_data_R{row['row']}.json")
        row_args.frame_drop_path = os.path.join(row_folder, f"frame_drop_{clip_id}.log")

        return row_args


def create_str_from_row(row):
    unique_str = str(row['customer_code']) + '_' + str(row['plot_code']) + '_' + str(
        row['scan_date']) + '_' + str(row['row']) + '_' + str(
        int(row['file_index']))
    return unique_str