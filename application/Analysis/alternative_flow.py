import traceback
import logging
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
    def init_module(qu, main_pid, module_name, communication_queue):
        super(AlternativeFlow, AlternativeFlow).init_module(qu, main_pid, module_name, communication_queue)
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

        collected, analyzed = AlternativeFlow.read_collected_analyzed()

        while True:
            found, row, row_index = AlternativeFlow.seek_new_row(collected, analyzed)
            if found:
                # using try in case of collapse in analysis flow
                try:
                    logging.info(f"Analyzing new Row: {list(row)}")
                    print(f'Analyzing new Row: {list(row)}')
                    row_runtime_args = AlternativeFlow.update_runtime_args(runtime_args, row)
                    rc = run(pipeline_conf, row_runtime_args)
                    print(f'Done analyzing row: {list(row)}')
                    is_success = True  # analysis ended without exceptions
                    data = AlternativeFlow.prepare_data(tracks=rc.tracks,
                                                        tracks_header=rc.tracks_header,
                                                        alignment=rc.alignment,
                                                        alignment_header=rc.alignment_header,
                                                        row=row,
                                                        status=is_success)

                    # send results to data manager
                    print("sending data from analysis: ", time.time())
                    AlternativeFlow.send_data(ModuleTransferAction.ANALYZED_DATA, data, ModulesEnum.DataManager)
                    logging.info(f"Done analyzing {list(row)}")
                except:
                    is_success = False
                    logging.exception(f"Failed to analyze {list(row)}")
                    print(f"Failed to analyze {list(row)}")
                    data = AlternativeFlow.prepare_data([], [], [], [], row, is_success)
                    # send results to data manager
                    AlternativeFlow.send_data(ModuleTransferAction.ANALYZED_DATA, data, ModulesEnum.DataManager)
                finally:
                    collected.drop(index=row_index, inplace=True)
            else:
                logging.info('No new file found, waiting 1 minute')
                print('No new file found, waiting 1 minute')
                time.sleep(60)
                collected, analyzed = AlternativeFlow.read_collected_analyzed()

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
            if unique_str not in analyzed_list:
                found_new_row = True
                break

        return found_new_row, row, row_index

    @staticmethod
    def update_runtime_args(args, row):

        row_args = args.copy()
        folder_index = str(int(row['folder_index']))
        # toDo: add ext to collected to understand ext
        #ext = row['ext']
        ext = 'csv'

        row_folder = os.path.join(data_conf.output_path, row['customer_code'], row['plot_code'],
                                  str(row['scan_date']), f"row_{int(row['row'])}", folder_index)
        row_args.output_folder = row_folder
        row_args.row_name = int(row['row'])
        row_args.zed.movie_path = os.path.join(row_folder, f"ZED.mkv")
        row_args.depth.movie_path = os.path.join(row_folder, f"DEPTH.mkv")
        row_args.jai.movie_path = os.path.join(row_folder, f"Result_FSI.mkv")
        row_args.rgb_jai.movie_path = os.path.join(row_folder, f"Result_RGB.mkv")
        row_args.sync_data_log_path = os.path.join(row_folder, f"{data_conf.jaized_timestamps}.{ext}")
        row_args.slice_data_path = os.path.join(row_folder, f"Result_FSI_slice_data_R{row['row']}.json")
        row_args.frame_drop_path = os.path.join(row_folder, f"frame_drop.log")

        print(row_args.row_name)
        print(row_args.zed.movie_path)
        print(row_args.sync_data_log_path)

        return row_args

    @staticmethod
    def prepare_data(tracks, tracks_header, alignment, alignment_header, row, status):
        data = {
            'tracks': np.array(tracks),
            'tracks_headers': tracks_header,
            'alignment': np.array(alignment),
            'alignment_headers': alignment_header,
            'row': row,
            'status': status
        }

        return data

    @staticmethod
    def read_collected_analyzed():
        while True:
            try:
                collected = pd.read_csv(data_conf.collected_path, dtype=str)
                print('collected was read')
                break
            except (FileNotFoundError, PermissionError):
                print('collected not read')
                time.sleep(60)
                pass
            except Exception:
                traceback.print_exc()
        try:
            analyzed = pd.read_csv(data_conf.analyzed_path, dtype=str)
        except (FileNotFoundError, PermissionError):
            analyzed = pd.DataFrame()
        return collected, analyzed


def create_str_from_row(row):
    unique_str = '_'.join([str(row['customer_code']), str(row['plot_code']), str(row['scan_date']),
                          str(row['row']), str(int(row['folder_index']))])

    return unique_str


