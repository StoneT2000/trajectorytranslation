import json
import os
import os.path as osp
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import torch

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
import wandb as wb

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
        project_name: str = None,
        wandb_cfg = None,
        cfg: Union[Dict, OmegaConf] = {},
    ) -> None:
        """

        Parameters
        ----------

        clear_out : bool
            If true, clears out all previous logging information for this experiment. Otherwise appends data only
        """
        self.wandb = wandb
        if wandb_cfg is None:
            wandb_cfg = {}
        self.tensorboard = tensorboard

        self.exp_path = osp.join(workspace, exp_name)
        self.log_path = osp.join(self.exp_path, "logs")
        self.raw_log_file = osp.join(self.log_path, "raw.csv")
        if clear_out:
            if osp.exists(self.exp_path):
                shutil.rmtree(self.exp_path, ignore_errors=True)

        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        # set up external loggers
        if self.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=self.log_path)
        if self.wandb:
            
            if project_name is None:
                project_name = workspace
            if not clear_out:
                wandb_id = cfg["wandb_id"]
                wb.init(project=project_name, name=exp_name, id=wandb_id, resume="allow", **wandb_cfg)
            else:
                wandb_id = wb.util.generate_id()
                wb.init(project=project_name, name=exp_name, id=wandb_id, **wandb_cfg)
                cfg["wandb_id"] = wandb_id
        self.save_config(cfg)

        self.data = defaultdict(dict)
        self.stats = {}

    def close(self):
        """
        finishes up experiment logging

        in wandb, finishes the experiment and uploads remaining data
        """
        if self.tensorboard:
            self.tb_writer.close()
        if self.wandb:
            wb.finish()

    def save_config(self, config: Union[Dict, OmegaConf], verbose=2):
        """
        save configuration of experiments to the experiment directory
        """
        if type(config) == type(OmegaConf.create()):
            config = OmegaConf.to_container(config)
        if self.wandb:
            wb.config.update(config, allow_val_change=True)
        config_path = osp.join(self.exp_path, "config.yml")
        # config_json = convert_json(config)
        # output = json.dumps(config_json, indent=2, sort_keys=True)
        if verbose > 1:
            self.print("Saving config:\n", color="cyan", bold=True)
        if verbose > 1:
            self.print(config)
        with open(config_path, "w") as out:
            out.write(OmegaConf.to_yaml(config))

    def print(self, msg, file=sys.stdout, color="", bold=False):
        """
        print to terminal, stdout by default. Ensures only the main process ever prints.
        """
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
        vals = []
        key_lens = [len(key) for key in data.keys()]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in sorted(data.keys()):
            val = data[key]
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            vals.append(val)
        print("-" * n_slashes, flush=True)

    def log(self, step, local_only=False):
        """
        log accumulated data to tensorboard if enabled and to the terminal and locally. Also syncs collected data across processes

        Statistics are then retrievable as a dict via get_statistics

        """
        for tag in self.data.keys():
            data_dict = self.data[tag]
            for k, v in data_dict.items():
                if isinstance(v, list):
                    vals = np.array(v)
                    vals_sum, n = vals.sum(), len(vals)
                    avg = vals_sum / n
                    sum_sq = np.sum((vals - avg) ** 2)
                    std = np.sqrt(sum_sq / n)
                    minv = np.min(vals)
                    maxv = np.max(vals)
                    key_vals = {
                        f"{tag}/{k}_avg": avg,
                        f"{tag}/{k}_std": std,
                        f"{tag}/{k}_min": minv,
                        f"{tag}/{k}_max": maxv,
                    }
                else:
                    key_vals = {f"{tag}/{k}": v}
                for name, scalar in key_vals.items():
                    if self.tensorboard and not local_only:
                        self.tb_writer.add_scalar(name, scalar, step)
                    self.stats[name] = scalar
                if self.wandb and not local_only:
                    wb.log(data=key_vals, step=step)
                    
        return self.stats

    def reset(self):
        """
        call this each time after log is called
        """
        self.data = defaultdict(dict)
        self.stats = {}
