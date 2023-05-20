import traceback

import pandas as pd
import os
import time

from application.utils.module_wrapper import ModulesEnum, Module, ModuleTransferAction
from application.utils.settings import pipeline_conf, runtime_args, data_conf
from vision.pipelines.adt_pipeline import run
import signal
import numpy as np

class AlternativeFlow(Module):

    @staticmethod
    def init_module(qu, main_pid, module_name):
        super(AlternativeFlow, AlternativeFlow).init_module(qu, main_pid, module_name)
        signal.signal(signal.SIGTERM, AlternativeFlow.shutdown)
        signal.signal(signal.SIGUSR1, AlternativeFlow.receive_data)
        AlternativeFlow.analyze()
        print("Analyze process is up")

    @staticmethod
    def shutdown(sig, frame):
        pass

    @staticmethod
    def receive_data(sig, frame):
        pass

    @staticmethod
    def analyze():

        def read_collected_analyzed():
            while True:
                try:
                    collected = pd.read_csv(data_conf['collected path'])
                    print('collected was read')
                    break
                except (FileNotFoundError, PermissionError):
                    print('collected not read')
                    pass
                except Exception:
                    traceback.print_exc()
            try:
                analyzed = pd.read_csv(data_conf['analyzed path'])
            except (FileNotFoundError, PermissionError):
                analyzed = pd.DataFrame()
            return collected, analyzed

        collected, analyzed = read_collected_analyzed()

        while True:
            found, row, row_index = AlternativeFlow.seek_new_row(collected, analyzed)
            if found:
                collected.drop(index=row_index, inplace=True)
                print(f'Analyzing new Row: {list(row)}')
                row_runtime_args = AlternativeFlow.update_runtime_args(runtime_args, row)
                rc = run(pipeline_conf, row_runtime_args)
                print(f'Done analyzing row: {list(row)}')
                tracks = np.array(rc.tracks)
                alignment = np.array(rc.alignment)
                data = {
                    'tracks': tracks,
                    'tracks_headers': rc.tracks_header,
                    'alignment': alignment,
                    'alignment_headers': rc.alignment_header,
                    'row': row
                }
                #send results to data manager
                # todo change ModuleTransferAction value
                print("sending data from analysis: ", time.time())
                AlternativeFlow.send_data(ModuleTransferAction.ANALYZED_DATA, data, ModulesEnum.DataManager)
            else:
                print('No new file Found, waiting 1 minute')
                collected, analyzed = read_collected_analyzed()
            time.sleep(60)

    @staticmethod
    def seek_new_row(collected, analyzed):
        analyzed_list = []
        for k, row in analyzed.iterrows():
            analyzed_list.append(create_str_from_row(row))

        row = None  # in case collected is empty - argument sent in return
        row_index = None
        found_new_row = False
        for row_index, row in collected.iterrows():
            unique_str = create_str_from_row(row)
            if unique_str in analyzed_list:
                continue
            else:
                found_new_row = True
                break

        return found_new_row, row, row_index


    @staticmethod
    def update_runtime_args(args, row):

        row_args = args.copy()
        row_folder = os.path.join(data_conf['output path'], row['customer_code'], row['plot_code'],
                                  str(row['scan_date']), f"row_{int(row['row'])}")
        clip_id = int(row['file_index'])
        row_args.output_folder = row_folder
        row_args.row_name = clip_id
        row_args.zed.movie_path = os.path.join(row_folder, f"ZED_{clip_id}.mkv")
        row_args.depth.movie_path = os.path.join(row_folder, f"DEPTH_{clip_id}.mkv")
        row_args.jai.movie_path = os.path.join(row_folder, f"FSI_EQUALIZE_HIST_{clip_id}.mkv")
        row_args.rgb_jai.movie_path = os.path.join(row_folder, f"RGB_{clip_id}.mkv")
        row_args.sync_data_log_path = os.path.join(row_folder, f"jaized_timestamps_{clip_id}.log")
        row_args.slice_data_path = os.path.join(row_folder, f"Result_FSI_{clip_id}_slice_data_R{row['row']}.json")
        row_args.frame_drop_path = os.path.join(row_folder, f"frame_drop_{clip_id}.log")

        return row_args


def create_str_from_row(row):
    unique_str = str(row['customer_code']) + '_' + str(row['plot_code']) + '_' + str(
        row['scan_date']) + '_' + str(row['row']) + '_' + str(
        int(row['file_index']))
    return unique_str