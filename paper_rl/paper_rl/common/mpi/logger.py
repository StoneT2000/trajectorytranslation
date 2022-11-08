"""

Some simple logging functionality, inspired by rllab's logging.

Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

"""
import atexit
import json
import os
import os.path as osp
import shutil
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch
from box_pusher.agents.utils.mpi_tools import mpi_statistics_scalar, proc_id
from box_pusher.agents.utils.serialization_utils import convert_json
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


class Logger:
    def __init__(
        self, workspace: str, exp_name="default", tensorboard=True, clear_out=False
    ) -> None:

        # self.data_dict = defaultdict(list)
        self.tagged_data = {}
        self.raw_values_keys = (
            set()
        )  # set of keys for values that don't need statistics computed
        self.stats = {}
        self.tb_writer: SummaryWriter = None
        self.tensorboard = tensorboard
        self.workspace = workspace
        self.exp_path = osp.join(workspace, exp_name)
        self.clear_out = clear_out
        self.log_path = osp.join(self.exp_path, "logs")
        self.model_path = osp.join(self.exp_path, "models")
        self.raw_log_file = osp.join(self.exp_path, "raw.csv")
        self.headers = []

        if proc_id() == 0:
            Path(self.workspace).mkdir(parents=True, exist_ok=True)
            if clear_out:
                if osp.exists(self.exp_path):
                    shutil.rmtree(self.exp_path, ignore_errors=True)
            Path(self.exp_path).mkdir(parents=True, exist_ok=True)
            Path(self.model_path).mkdir(parents=True, exist_ok=True)
            if self.tensorboard:
                self.tb_writer = SummaryWriter(log_dir=self.log_path)

    def close(self):
        if proc_id() == 0 and self.tb_writer is not None:
            self.tb_writer.close()

    def setup_pytorch_saver(self, model):
        """
        setup saver so logger has a reference to what needs to be saved. Makeslogger a little more efficient and avoids the caller having to deal with
        proc ids
        """
        self.model = model

    def save_config(self, config: Dict, verbose=2):
        """
        save configuration of experiments to the experiment directory
        """
        if proc_id() == 0:
            config_path = osp.join(self.exp_path, "config.json")
            config_json = convert_json(config)
            output = json.dumps(config_json, indent=2, sort_keys=True)
            if verbose > 1:
                self.print("Saving config:\n", color="cyan", bold=True)
            if verbose > 1:
                self.print(output)
            with open(config_path, "w") as out:
                out.write(output)

    def save_model(self, name):
        """
        save the model
        """
        if proc_id() == 0:
            torch.save(self.model.state_dict(), osp.join(self.model_path, name))

    def print(self, msg, file=sys.stdout, color="", bold=False):
        """
        print to terminal, stdout by default. Ensures only the main process ever prints.
        """
        if proc_id() == 0:
            if color == "":
                print(msg, file=file)
            else:
                print(colorize(msg, color, bold=bold), file=file)
            sys.stdout.flush()

    def store(self, tag="default", value_only=False, **kwargs):
        """
        store some scalar value to a key, which is accumulated until logged.

        if value_only is True, then when printing/logging this data, no statistics aggregation is done. Expect only one worker to ever call store with value_only=True
        """
        if tag not in self.tagged_data:
            self.tagged_data[tag] = defaultdict(list)
        data_dict = self.tagged_data[tag]
        for k, v in kwargs.items():
            data_dict[k].append(v)
            if value_only == True:
                self.raw_values_keys.add(f"{tag}/{k}")

    def get_statistics(self):
        return self.stats

    def pretty_print_table(self, data):
        if proc_id() == 0:
            vals = []
            key_lens = [len(key) for key in data.keys()]
            max_key_len = max(15, max(key_lens))
            keystr = "%" + "%d" % max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            print("-" * n_slashes)
            for key in data.keys():
                val = data[key]
                valstr = "%8.3g" % val if hasattr(val, "__float__") else val
                print(fmt % (key, valstr))
                vals.append(val)
            print("-" * n_slashes, flush=True)

    def log(self, step):
        """
        log accumulated data to tensorboard if enabled and to the terminal and locally. Also syncs collected data across processes

        Statistics are then retrievable as a dict via get_statistics

        """
        # if val is not None:
        #     super().log_tabular(key, val)
        # else:
        for tag in self.tagged_data.keys():
            data_dict = self.tagged_data[tag]
            for k, v in data_dict.items():
                vals = (
                    np.concatenate(v)
                    if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
                    else v
                )
                if f"{tag}/{k}" not in self.raw_values_keys:
                    stats = mpi_statistics_scalar(vals, with_min_and_max=True)
                    avg, std, minv, maxv = stats[0], stats[1], stats[2], stats[3]
                    key_vals = {
                        f"{tag}/{k}_avg": avg,
                        f"{tag}/{k}_std": std,
                        f"{tag}/{k}_min": minv,
                        f"{tag}/{k}_max": maxv,
                    }
                else:
                    if isinstance(v, list):
                        if len(v) == 1:
                            vals = v[0]
                        else:
                            vals = np.array(v)
                    key_vals = {
                        f"{tag}/{k}": vals,
                    }
                if proc_id() == 0:
                    for name, scalar in key_vals.items():
                        if self.tensorboard:
                            self.tb_writer.add_scalar(name, scalar, step)
                        self.stats[name] = scalar
        if proc_id() == 0:
            if not osp.isfile(self.raw_log_file):
                with open(self.raw_log_file, "w") as f:
                    self.headers = []
                    for h in sorted(list(self.stats.keys())):
                        self.headers.append(h)
                    f.write(",".join(self.headers) + "\n")
            new_headers = False
            for k in self.stats.keys():
                if k not in self.headers:
                    self.headers.append(k)
                    new_headers = True
            if new_headers:
                os.rename(self.raw_log_file, self.raw_log_file + ".temp")
                orig_contents = []
                with open(self.raw_log_file + ".temp", "r") as f:
                    orig_contents = f.readlines()
                with open(self.raw_log_file, "w") as f:
                    f.write(",".join(self.headers) + "\n")
                    f.write("".join(orig_contents[1:]))
                os.remove(self.raw_log_file + ".temp")
            with open(self.raw_log_file, "a") as f:
                vals = []
                for h in self.headers:
                    if h in self.stats:
                        vals.append(str(self.stats[h]))
                    else:
                        vals.append("")
                f.write(",".join(vals) + "\n")

    def reset(self):
        """
        call this each time after log is called
        """
        for tag in self.tagged_data.keys():
            self.tagged_data[tag] = defaultdict(list)
        self.stats = {}
