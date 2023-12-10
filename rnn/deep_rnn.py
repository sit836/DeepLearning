import torch
from d2l import torch as d2l
from torch import nn


class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = nn.Sequential(
            *[d2l.RNNScratch(num_inputs if i == 0 else num_hiddens, num_hiddens, sigma) for i in range(num_layers)])

    def forward(self, inputs, Hs=None):
        outputs = inputs
        if Hs is None: Hs = [None] * self.num_layers
        for i in range(self.num_layers):
            outputs, Hs[i] = self.rnns[i](outputs, Hs[i])
            outputs = torch.stack(outputs, 0)
        return outputs, Hs


batch_size = 10
num_steps = 5
num_hiddens = 32
num_layers = 2
data = d2l.TimeMachine(batch_size=batch_size, num_steps=num_steps)
rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                              num_hiddens=num_hiddens, num_layers=num_layers)
model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
trainer = d2l.Trainer(max_epochs=3, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
