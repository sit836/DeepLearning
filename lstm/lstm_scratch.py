import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l
from torch import nn


class LSTMScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(num_inputs, num_hiddens),
                          init_weight(num_hiddens, num_hiddens),
                          nn.Parameter(torch.zeros(num_hiddens)))
        self.W_xi, self.W_hi, self.b_i = triple()  # Input gate
        self.W_xf, self.W_hf, self.b_f = triple()  # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple()  # Output gate
        self.W_xc, self.W_hc, self.b_c = triple()  # Input node

    def forward(self, inputs, H_C=None):
        if H_C is None:
            # Initial state with shape: (batch_size, num_hiddens)
            H = torch.zeros((inputs.shape[1], self.num_hiddens),
                            device=inputs.device)
            C = torch.zeros((inputs.shape[1], self.num_hiddens),
                            device=inputs.device)
        else:
            H, C = H_C
        outputs = []

        for X in inputs:
            I = torch.sigmoid(torch.matmul(X, self.W_xi) +
                              torch.matmul(H, self.W_hi) + self.b_i)
            F = torch.sigmoid(torch.matmul(X, self.W_xf) +
                              torch.matmul(H, self.W_hf) + self.b_f)
            O = torch.sigmoid(torch.matmul(X, self.W_xo) +
                              torch.matmul(H, self.W_ho) + self.b_o)
            C_tilde = torch.tanh(torch.matmul(X, self.W_xc) +
                                 torch.matmul(H, self.W_hc) + self.b_c)
            C = F * C + I * C_tilde
            H = O * torch.tanh(C)
            outputs.append(H)
        return outputs, (H, C)


if __name__ == "__main__":
    lr = 4
    batch_size, num_inputs, num_hiddens, num_steps = 1024, 4, 3, 10
    data = d2l.TimeMachine(batch_size=batch_size, num_steps=num_steps)
    lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=num_hiddens)
    model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=lr)
    trainer = d2l.Trainer(max_epochs=5, gradient_clip_val=1, num_gpus=1)
    trainer.fit(model, data)
    plt.show()
