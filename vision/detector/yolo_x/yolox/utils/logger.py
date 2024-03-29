#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import inspect
import os
import sys
from collections import defaultdict
from loguru import logger

import cv2
import numpy as np
import pandas as pd
import sklearn.utils as sklearn_utils

import torch


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth.
        Default value: 0.

    Returns:
        str: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals["__name__"]


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """

    def __init__(self, level="INFO", caller_names=("apex", "pycocotools")):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        pass

    def isatty(self):
        # when using colab, jax is installed by default and issue like
        # https://github.com/Megvii-BaseDetection/YOLOX/issues/1437 might be raised
        # due to missing attribute like`isatty`.
        # For more details, checked the following link:
        # https://github.com/google/jax/blob/10720258ea7fb5bde997dfa2f3f71135ab7a6733/jax/_src/pretty_printer.py#L54  # noqa
        return True


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(
            sys.stderr,
            format=loguru_format,
            level="INFO",
            enqueue=True,
        )
        logger.add(save_file)

    # redirect stdout/stderr to loguru
    redirect_sys_output("INFO")


class WandbLogger(object):
    """
    Log training runs, datasets, models, and predictions to Weights & Biases.
    This logger sends information to W&B at wandb.ai.
    By default, this information includes hyperparameters,
    system configuration and metrics, model metrics,
    and basic data metrics and analyses.

    For more information, please refer to:
    https://docs.wandb.ai/guides/track
    https://docs.wandb.ai/guides/integrations/other/yolox
    """
    def __init__(self,
                 project=None,
                 name=None,
                 id=None,
                 entity=None,
                 save_dir=None,
                 config=None,
                 val_dataset=None,
                 num_eval_images=0,  # 100,
                 log_checkpoints=False,
                 **kwargs):
        """
        Args:
            project (str): wandb project name.
            name (str): wandb run name.
            id (str): wandb run id.
            entity (str): wandb entity name.
            save_dir (str): save directory.
            config (dict): config dict.
            val_dataset (Dataset): validation dataset.
            num_eval_images (int): number of images from the validation set to log.
            log_checkpoints (bool): log checkpoints
            **kwargs: other kwargs.

        Usage:
            Any arguments for wandb.init can be provided on the command line using
            the prefix `wandb-`.
            Example
            ```
            python tools/train.py .... --logger wandb wandb-project <project-name> \
                wandb-name <run-name> \
                wandb-id <run-id> \
                wandb-save_dir <save-dir> \
                wandb-num_eval_imges <num-images> \
                wandb-log_checkpoints <bool>
            ```
            The val_dataset argument is not open to the command line.
        """
        try:
            import wandb
            self.wandb = wandb
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "wandb is not installed."
                "Please install wandb using pip install wandb"
                )

        self.project = project
        self.name = name
        self.id = id
        self.save_dir = save_dir
        self.config = config
        self.kwargs = kwargs
        self.entity = entity
        self._run = None
        self.val_artifact = None
        self.num_eval_images = num_eval_images
        if num_eval_images == -1:
            self.num_log_images = len(val_dataset)
        else:
            self.num_log_images = min(num_eval_images, len(val_dataset))
        self.log_checkpoints = (log_checkpoints == "True" or log_checkpoints == "true")
        self._wandb_init = dict(
            project=self.project,
            name=self.name,
            id=self.id,
            entity=self.entity,
            dir=self.save_dir,
            resume="allow"
        )
        self._wandb_init.update(**kwargs)

        _ = self.run

        if self.config:
            self.run.config.update(self.config)
        self.run.define_metric("train/epoch")
        self.run.define_metric("val/*", step_metric="train/epoch")
        self.run.define_metric("train/step")
        self.run.define_metric("train/*", step_metric="train/step")

        if val_dataset and self.num_log_images != 0:
            self.cats = val_dataset.cats
            self.id_to_class = {
                cls['id']: cls['name'] for cls in self.cats
            }
            self._log_validation_set(val_dataset)

    @property
    def run(self):
        if self._run is None:
            if self.wandb.run is not None:
                logger.info(
                    "There is a wandb run already in progress "
                    "and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()`"
                    "before instantiating `WandbLogger`."
                )
                self._run = self.wandb.run
            else:
                self._run = self.wandb.init(**self._wandb_init)
        return self._run

    def _log_validation_set(self, val_dataset):
        """
        Log validation set to wandb.

        Args:
            val_dataset (Dataset): validation dataset.
        """
        if self.val_artifact is None:
            self.val_artifact = self.wandb.Artifact(name="validation_images", type="dataset")
            self.val_table = self.wandb.Table(columns=["id", "input"])

            for i in range(self.num_log_images):
                data_point = val_dataset[i]
                img = data_point[0]
                id = data_point[3]
                img = np.transpose(img, (1, 2, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.val_table.add_data(
                    id.item(),
                    self.wandb.Image(img)
                )

            self.val_artifact.add(self.val_table, "validation_images_table")
            self.run.use_artifact(self.val_artifact)
            self.val_artifact.wait()

    def log_metrics(self, metrics, step=None):
        """
        Args:
            metrics (dict): metrics dict.
            step (int): step number.
        """

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v.item()

        if step is not None:
            metrics.update({"train/step": step})
            self.run.log(metrics)
        else:
            self.run.log(metrics)

    def log_images(self, predictions):
        if len(predictions) == 0 or self.val_artifact is None or self.num_log_images == 0:
            return

        table_ref = self.val_artifact.get("validation_images_table")

        columns = ["id", "predicted"]
        for cls in self.cats:
            columns.append(cls["name"])

        result_table = self.wandb.Table(columns=columns)
        for idx, val in table_ref.iterrows():

            avg_scores = defaultdict(int)
            num_occurrences = defaultdict(int)

            if val[0] in predictions:
                prediction = predictions[val[0]]
                boxes = []

                for i in range(len(prediction["bboxes"])):
                    bbox = prediction["bboxes"][i]
                    x0 = bbox[0]
                    y0 = bbox[1]
                    x1 = bbox[2]
                    y1 = bbox[3]
                    box = {
                        "position": {
                            "minX": min(x0, x1),
                            "minY": min(y0, y1),
                            "maxX": max(x0, x1),
                            "maxY": max(y0, y1)
                        },
                        "class_id": prediction["categories"][i],
                        "domain": "pixel"
                    }
                    avg_scores[
                        self.id_to_class[prediction["categories"][i]]
                    ] += prediction["scores"][i]
                    num_occurrences[self.id_to_class[prediction["categories"][i]]] += 1
                    boxes.append(box)
            else:
                boxes = []

            average_class_score = []
            for cls in self.cats:
                if cls["name"] not in num_occurrences:
                    score = 0
                else:
                    score = avg_scores[cls["name"]] / num_occurrences[cls["name"]]
                average_class_score.append(score)
            result_table.add_data(
                idx,
                self.wandb.Image(val[1], boxes={
                        "prediction": {
                            "box_data": boxes,
                            "class_labels": self.id_to_class
                        }
                    }
                ),
                *average_class_score
            )

        self.wandb.log({"val_results/result_table": result_table})

    def save_checkpoint(self, save_dir, model_name, is_best, metadata=None):
        """
        Args:
            save_dir (str): save directory.
            model_name (str): model name.
            is_best (bool): whether the model is the best model.
            metadata (dict): metadata to save corresponding to the checkpoint.
        """

        if not self.log_checkpoints:
            return

        if "epoch" in metadata:
            epoch = metadata["epoch"]
        else:
            epoch = None

        filename = os.path.join(save_dir, model_name + "_ckpt.pth")
        artifact = self.wandb.Artifact(
            name=f"run_{self.run.id}_model",
            type="model",
            metadata=metadata
        )
        artifact.add_file(filename, name="model_ckpt.pth")

        aliases = ["latest"]

        if is_best:
            aliases.append("best")

        if epoch:
            aliases.append(f"epoch-{epoch}")

        self.run.log_artifact(artifact, aliases=aliases)

    def finish(self):
        self.run.finish()

    @classmethod
    def initialize_wandb_logger(cls, args, exp, val_dataset):
        wandb_params = dict()
        prefix = "wandb-"
        for k, v in zip(args.opts[0::2], args.opts[1::2]):
            if k.startswith("wandb-"):
                try:
                    wandb_params.update({k[len(prefix):]: int(v)})
                except ValueError:
                    wandb_params.update({k[len(prefix):]: v})

        return cls(config=vars(exp), val_dataset=val_dataset, **wandb_params)

    @staticmethod
    def eval_to_df(eval, iou=0.5):
        params = eval['params']
        iou_thrs = list(params.iouThrs)
        recall_list = list(params.recThrs)
        cat_ids = list(params.catIds)
        #######
        if min(cat_ids) == 1:                    # fixed a bug if cat_ids start at 1 and not 0. todo - check if not disturbing other things
            cat_ids = [cat_id - 1 for cat_id in cat_ids]
        #####
        iou_ind = iou_thrs.index(iou)

        res_len = len(recall_list)
        data_list = []
        for cat_id in cat_ids:
            if -1 in eval['precision'][iou_ind, :, cat_id, 0, -1]:
                continue
            else:
                for i in range(res_len):

                    precision = eval['precision'][iou_ind, i, cat_id, 0, -1]
                    score = eval['scores'][iou_ind, i, cat_id, 0, -1]
                    recall = recall_list[i]

                    data_list.append({'precision': precision, "recall": recall, "score": score, 'class': cat_id})

        return pd.DataFrame(data=data_list, columns=['precision', 'recall', 'score', 'class'])


    def eval_to_table(self, eval, iou=0.5):

        df = self.eval_to_df(eval, iou)
        df = df.round(3)

        if len(df) > self.wandb.Table.MAX_ROWS:
            self.wandb.termwarn(
                "wandb uses only %d data points to create the plots." % self.wandb.Table.MAX_ROWS
            )
            # different sampling could be applied, possibly to ensure endpoints are kept
            df = sklearn_utils.resample(
                df,
                replace=False,
                n_samples=self.wandb.Table.MAX_ROWS,
                random_state=42,
                stratify=df["class"],
            ).sort_values(["precision", "recall", "score", "class"])

        return self.wandb.Table(dataframe=df)

    def log_pr_table(self, eval, iou=0.5):

        pr_table = self.eval_to_table(eval, iou)
        self.run.log({"pr_table": pr_table})

    def log_f1_table(self, eval, iou=0.5):

        # Convert eval metrics to a pandas DataFrame
        df = self.eval_to_df(eval, iou)

        # Calculate F1 scores
        df['f1'] = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'] + 1e-16)

        # Round to three decimal places
        df = df.round(3)

        # Make sure we don't exceed wandb's maximum row limit
        if len(df) > self.wandb.Table.MAX_ROWS:
            self.wandb.termwarn(
                "wandb uses only %d data points to create the plots." % self.wandb.Table.MAX_ROWS
            )
            df = sklearn_utils.resample(
                df,
                replace=False,
                n_samples=self.wandb.Table.MAX_ROWS,
                random_state=42,
                stratify=df["class"]
            ).sort_values(["score"])

        # Log the F1 table including 'score'
        f1_table = self.wandb.Table(dataframe=df)
        self.run.log({"f1_table": f1_table})

        # Find the score with best f1:
        max_f1_index = df['f1'].idxmax()
        best_score = df.loc[max_f1_index, 'score']
        best_f1 = df.loc[max_f1_index, 'f1']

        # Prepare the data for plotting
        score_values = df['score'].tolist()  # Score values
        f1_scores = df['f1'].tolist()  # F1 score values

        # Log the F1 plot
        self.run.log({
            "f1_curve": self.wandb.plot.line_series(
                xs=[score_values],  # Score values wrapped in a list
                ys=[f1_scores],  # F1 score values wrapped in a list
                keys=["F1 Score"],
                title=f"F1 vs Conf Threshold. Best conf {best_score}",
                xname="Confidence threshold"
            )
        })

        # Print the best score and corresponding F1 score
        print(f"Best F1 Score: {best_f1} at Confidence threshold: {best_score}")


    def log_train_artifact(self, artifact_name, weights_file_path, exp_path, run_log_path):
        artifact = self.wandb.Artifact(artifact_name, type='model')
        artifact.add_file(weights_file_path)
        artifact.add_file(exp_path)
        artifact.add_file(run_log_path)

        self.run.log_artifact(artifact, aliases=['latest'])