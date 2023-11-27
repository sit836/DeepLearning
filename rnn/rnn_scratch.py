import torch
from torch import nn

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
if __name__ == "__main__":
    batch_size, num_inputs, num_hiddens, num_steps = 2, 4, 3, 10
    rnn = RNNScratch(num_inputs, num_hiddens)
    X = torch.ones((num_steps, batch_size, num_inputs))
    outputs, state = rnn(X)
    print(f'outputs: {outputs}')
    print(f'X.shape: {X.shape}')
    print(f'state.shape: {state.shape}')
    print(f'rnn.W_xh.shape: {rnn.W_xh.shape}')
    print(f'rnn.W_hh.shape: {rnn.W_hh.shape}')
    print(f'rnn.b_h.shape: {rnn.b_h.shape}')
