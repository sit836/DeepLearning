import torch
from torch import nn
from torch.nn import functional as F

from rnn_scratch import RNNScratch


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
        print(f'X: {X}')
        embs = self.one_hot(X)
        print(f'embs: {embs}')
        quit()

        rnn_outputs, _ = self.rnn(embs, state)
        return self.output_layer(rnn_outputs)


if __name__ == "__main__":
    batch_size, num_inputs, num_hiddens, num_steps = 2, 4, 3, 5
    rnn = RNNScratch(num_inputs, num_hiddens)

    model = RNNLMScratch(rnn, vocab_size=num_inputs)
    outputs = model(torch.ones((batch_size, num_steps), dtype=torch.int64))
    print('\n\n')
    print(f'batch_size, num_inputs, num_hiddens, num_steps: {batch_size, num_inputs, num_hiddens, num_steps}')
    print(f'outputs: {outputs}')
    print(f'outputs.shape: {outputs.shape}')
