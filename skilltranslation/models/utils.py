import torch.nn as nn
def act2fnc(act_name):
    if act_name == "relu":
        return nn.ReLU
    if act_name == "tanh":
        return nn.Tanh
    if act_name == "gelu":
        return nn.GELU
    if act_name == "identity" or act_name == "":
        return nn.Identity