import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self, out_dims, input_shape) -> None:
        super().__init__()
        self.out_dims = out_dims
        self.input_shape = input_shape