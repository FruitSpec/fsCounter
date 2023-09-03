import traceback
import logging
import pandas as pd
import os
import time
import threading

from application.utils.module_wrapper import ModulesEnum, Module, ModuleTransferAction
from application.utils.settings import pipeline_conf, runtime_args, data_conf, conf
from vision.pipelines.adt_pipeline import run
import signal
import numpy as np


class AlternativeFlow(Module):

    @staticmethod
    def init_module(in_qu, out_qu, main_pid, module_name, communication_queue, notify_on_death, death_action):
        super(AlternativeFlow, AlternativeFlow).init_module(in_qu, out_qu, main_pid, module_name,
                                                            communication_queue, notify_on_death, death_action)
        signal.signal(signal.SIGTERM, AlternativeFlow.shutdown)
        AlternativeFlow.receive_data_thread = threading.Thread(target=AlternativeFlow.receive_data, daemon=True)

        AlternativeFlow.receive_data_thread.start()

        AlternativeFlow.analyze()
        AlternativeFlow.receive_data_thread.join()

    @staticmethod
    def receive_data():
        pass

    @staticmethod
    def analyze():

        collected, analyzed = AlternativeFlow.read_collected_analyzed()

        while True:
            found, row, row_index = AlternativeFlow.seek_new_row(collected, analyzed)
            if found:
                # AlternativeFlow.send_data(ModuleTransferAction.ANALYSIS_ONGOING, None, ModulesEnum.GPS)
                # using try in case of collapse in analysis flow
                try:
                    logging.info(f"ANALYZING NEW ROW: {list(row)}")
                    print(f"ANALYZING NEW ROW: {list(row)}")

                    row_runtime_args = AlternativeFlow.update_runtime_args(runtime_args, row)
                    rc = run(pipeline_conf, row_runtime_args)
                    logging.info(f"DONE ANALYZING ROW: {list(row)}")
                    print(f"DONE ANALYZING ROW: {list(row)}")
                    is_success = True  # analysis ended without exceptions

                    if conf.crop == "citrus":
                        data = AlternativeFlow.prepare_data(
                            tracks=rc.tracks,
                            tracks_header=rc.tracks_header,
                            alignment=rc.alignment,
                            alignment_header=rc.alignment_header,
                            jai_translation=rc.jai_translation,
                            jai_translation_header=rc.jai_translation_header,
                            row=row,
                            status=is_success
                        )
                    else:
                        data = AlternativeFlow.prepare_data(
                            tracks=rc.tracks,
                            tracks_header=rc.tracks_header,
                            alignment=rc.alignment,
                            alignment_header=rc.alignment_header,
                            row=row,
                            status=is_success
                        )

                    # send results to data manager
                    print("sending data from analysis: ", time.time())
                    AlternativeFlow.send_data(ModuleTransferAction.ANALYZED_DATA, data, ModulesEnum.DataManager)
                    logging.info(f"Done analyzing {list(row)}")
                except:
                    is_success = False
                    logging.exception(f"Failed to analyze {list(row)}")
                    print(f"Failed to analyze {list(row)}")
                    data = AlternativeFlow.prepare_data(row=row, status=is_success)
                    # send results to data manager
                    AlternativeFlow.send_data(ModuleTransferAction.ANALYZED_DATA, data, ModulesEnum.DataManager)
                finally:
                    collected.drop(index=row_index, inplace=True)
            else:
                # AlternativeFlow.send_data(ModuleTransferAction.ANALYSIS_DONE, None, ModulesEnum.GPS)
                logging.info('No new file found, waiting 1 minute')
                print('No new file found, waiting 1 minute')
                time.sleep(60)
                collected, analyzed = AlternativeFlow.read_collected_analyzed()

    @staticmethod
    def seek_new_row(collected, analyzed):
        analyzed_set = set()
        for k, row in analyzed.iterrows():
            if not is_valid_row(row):
                continue
            analyzed_set.add(create_str_from_row(row))

        row = None  # in case collected is empty - argument sent in return
        row_index = None
        found_new_row = False
        for row_index, row in collected.iterrows():
            if not is_valid_row(row):
                continue
            unique_str = create_str_from_row(row)
            if unique_str not in analyzed_set:
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


        return row_args

    @staticmethod
    def prepare_data(tracks=[], tracks_header=[], alignment=[], alignment_header=[], jai_translation=[],
                     jai_translation_header=[], row=-1, status=None):
        data = {
            'tracks': np.array(tracks),
            'tracks_header': tracks_header,
            'alignment': np.array(alignment),
            'alignment_header': alignment_header,
            'jai_translation': jai_translation,
            'jai_translation_header': jai_translation_header,
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

def is_valid_row(row):
    # check if any item in the row is NaN
    valid = True if np.sum(row.isna().to_numpy()) == 0 else False

    return valid