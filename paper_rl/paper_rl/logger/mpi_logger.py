import json
import os
import os.path as osp
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from paper_rl.common.mpi.mpi_tools import mpi_statistics_scalar, proc_id
from paper_rl.common.mpi.serialization_utils import convert_json

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


class MPILogger:
    """
    Logging tool
    """

    def __init__(
        self,
        wandb=False,
        tensorboard=False,
        workspace: str = "default_workspace",
        exp_name: str = "default_exp",
        clear_out: bool = True,
    ) -> None:
        """

        Parameters
        ----------

        clear_out : bool
            If true, clears out all previous logging information for this experiment. Otherwise appends data only
        """
        self.wandb = wandb
        self.tensorboard = tensorboard

        self.exp_path = osp.join(workspace, exp_name)
        self.log_path = osp.join(self.exp_path, "logs")
        self.raw_log_file = osp.join(self.log_path, "raw.csv")
        if proc_id() == 0:
            if clear_out:
                if osp.exists(self.exp_path):
                    shutil.rmtree(self.exp_path, ignore_errors=True)

        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        # set up external loggers
        if self.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter(log_dir=self.log_path)
        if self.wandb:
            pass

        self.data = defaultdict(dict)
        self.stats = {}

    def close(self):
        if self.tensorboard:
            self.tb_writer.close()

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

    def store(self, tag="default", append=True, **kwargs):
        """
        Stores scalar values into logger by tag and key to then be logged

        Parameters
        ----------
        append : bool
            If true, will append the value to a list of previously logged values under the same tag and key. If false, will
            replace what was stored previously. For the same tag and key, append should the same.

            When false, values are not aggregated across processes
        """
        for k, v in kwargs.items():
            if append:
                if k in self.data[tag]:
                    self.data[tag][k].append(v)
                else:
                    self.data[tag][k] = [v]
            else:
                self.data[tag][k] = v

    def get_data(self, tag=None):
        if tag is None:
            data_dict = {}
            for tag in self.data.keys():
                for k, v in self.data[tag].items():
                    data_dict[f"{tag}/{k}"] = v
            return data_dict
        return self.data[tag]

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
        for tag in self.data.keys():
            data_dict = self.data[tag]
            for k, v in data_dict.items():
                if isinstance(v, int) or isinstance(v, float):
                    key_vals = {f"{tag}/{k}": v}
                else:
                    vals = (
                        np.concatenate(v)
                        if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
                        else v
                    )
                    stats = mpi_statistics_scalar(vals, with_min_and_max=True)
                    avg, std, minv, maxv = stats[0], stats[1], stats[2], stats[3]
                    key_vals = {
                        f"{tag}/{k}_avg": avg,
                        f"{tag}/{k}_std": std,
                        f"{tag}/{k}_min": minv,
                        f"{tag}/{k}_max": maxv,
                    }
                if proc_id() == 0:
                    for name, scalar in key_vals.items():
                        if self.tensorboard:
                            self.tb_writer.add_scalar(name, scalar, step)
                        self.stats[name] = scalar
        return self.stats

    def reset(self):
        """
        call this each time after log is called
        """
        self.data = {}
        self.stats = {}
