import torch
from d2l import torch as d2l
from torch import nn


class PositionWiseFFN(nn.Module):
    """The positionwise feed-forward network."""

    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


batch_size = 2
num_steps = 3
num_hiddens = 4
ffn_num_hiddens, ffn_num_outputs = 4, 8
ffn = PositionWiseFFN(ffn_num_hiddens, ffn_num_outputs)
ffn.eval()
ffn(torch.ones((batch_size, num_steps, num_hiddens)))[0]
