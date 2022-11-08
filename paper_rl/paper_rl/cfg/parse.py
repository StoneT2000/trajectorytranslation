from omegaconf import DictConfig, OmegaConf
import os
import re
def parse_cfg(cfg_path: str = None, default_cfg_path: str = None) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object. Priority is CLI configs, then provided config, then the default config if it exists"""
    if default_cfg_path is not None:
        base = OmegaConf.load(default_cfg_path)
    else:
        base = OmegaConf.create()
   
    if cfg_path is not None:
        cfg = OmegaConf.load(cfg_path)
        base.merge_with(cfg)
    
    cli = OmegaConf.from_cli()
    for k, v in cli.items():
        if v is None:
            cli[k] = True
    base.merge_with(cli)
    return base
def clean_and_transform(cfg):
    for k, v in cfg.items():
        if isinstance(cfg[k], DictConfig): clean_and_transform(cfg[k])
        if isinstance(v, str):
            if v[0] == "(" and v[-1] == ")":
                cfg[k] = eval(v)
            elif v == "None":
                cfg[k] = None
                