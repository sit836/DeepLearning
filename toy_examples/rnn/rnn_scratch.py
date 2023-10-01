import math
import torch
from torch import nn
from torch.nn import functional as F

"""
    Dive into Deep Learning
"""


class RNNScratch(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super(RNNScratch, self).__init__()
        self.num_hiddens = num_hiddens
        self.sigma = sigma

        # nn.Parameter automatically register a class attribute as a parameter to be tracked by autograd
        self.W_xh = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.W_hh = nn.Parameter(torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))

    def forward(self, inputs, state=None):
        if state is None:
            # Initial state with shape: (batch_size, num_hiddens)
            state = torch.zeros((inputs.shape[1], self.num_hiddens), device=inputs.device)
        else:
            state = state

        outputs = []
        for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
            state = torch.tanh(torch.matmul(X, self.W_xh) + torch.matmul(state, self.W_hh) + self.b_h)
            outputs.append(state)
        return outputs, state


"""
    num_inputs: recall the conditional expectation E(X_t|X_t-1,...,X_1), where the number of inputs are X_t-1,...,X_1
"""
batch_size, num_inputs, num_hiddens, num_steps = 2, 4, 3, 10
rnn = RNNScratch(num_inputs, num_hiddens)
X = torch.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
print(f'batch_size, num_inputs, num_hiddens, num_steps: {batch_size, num_inputs, num_hiddens, num_steps}')
print(f'outputs: {outputs}')
print(f'X.shape: {X.shape}')
print(f'state.shape: {state.shape}')
print(f'rnn.W_xh.shape: {rnn.W_xh.shape}')
print(f'rnn.W_hh.shape: {rnn.W_hh.shape}')
print(f'rnn.b_h.shape: {rnn.b_h.shape}')


class RNNLMScratch(nn.Module):
    """The RNN-based language model implemented from scratch."""

    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.rnn = rnn
        self.vocab_size = vocab_size
        self.init_params()

    def init_params(self):
        self.W_hq = nn.Parameter(
            torch.randn(self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(torch.zeros(self.vocab_size))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', torch.exp(l), train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', torch.exp(l), train=False)

    def one_hot(self, X):
        # Output shape: (num_steps, batch_size, vocab_size)
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)

    def output_layer(self, rnn_outputs):
        outputs = [torch.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
        return torch.stack(outputs, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state)
        return self.output_layer(rnn_outputs)


model = RNNLMScratch(rnn, num_inputs)
outputs = model(torch.ones((batch_size, num_steps), dtype=torch.int64))
print('\n\n')
print(f'batch_size, num_inputs, num_hiddens, num_steps: {batch_size, num_inputs, num_hiddens, num_steps}')
print(f'outputs: {outputs}')
print(f'outputs.shape: {outputs.shape}')
