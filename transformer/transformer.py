import math

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


class AddNorm(nn.Module):
    """The residual connection followed by layer normalization."""

    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class TransformerEncoderBlock(nn.Module):
    """The Transformer encoder block."""

    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(d2l.Encoder):
    """The Transformer encoder."""

    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block" + str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)

        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X


class TransformerDecoderBlock(nn.Module):
    # The i-th block in the Transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)

        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every
            # row is [1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
            # print(f'state[2][self.i]: {state[2][self.i]}')
            print(f'key_values.shape: {key_values.shape}')

        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)  # Shape of Y: (batch_size, num_steps, num_hiddens)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block" + str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


if __name__ == "__main__":
    torch.manual_seed(0)

    # batch_size = 2
    # num_steps = 3
    # ffn_num_hiddens = 4
    # ffn_num_outputs = 8
    # num_hiddens = 5
    #
    # ffn = PositionWiseFFN(ffn_num_hiddens, ffn_num_outputs)
    # ffn.eval()
    #
    # #
    # ln = nn.LayerNorm(2)
    # bn = nn.LazyBatchNorm1d()
    # X = torch.tensor([[1, 2], [2, 300]], dtype=torch.float32)
    # # print('layer norm:', ln(X), '\nbatch norm:', bn(X))
    #
    # #
    # add_norm = AddNorm(num_hiddens, dropout=0.5)
    # shape = (batch_size, num_steps, num_hiddens)
    # d2l.check_shape(add_norm(torch.ones(shape), torch.ones(shape)), shape)
    #
    # #
    # batch_size, num_steps = 2, 100
    # num_hiddens, ffn_num_hiddens, num_heads = 24, 48, 8
    # X = torch.ones((batch_size, num_steps, num_hiddens))
    # valid_lens = torch.tensor([3, 2])
    # encoder_blk = TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout=0.5)
    # encoder_blk.eval()
    # d2l.check_shape(encoder_blk(X, valid_lens), X.shape)

    #
    # batch_size, num_steps = 3, 100
    # vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks = 200, 24, 48, 8, 2
    # valid_lens = torch.tensor([3] * batch_size)
    # encoder = TransformerEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout=0.5)
    # d2l.check_shape(encoder(torch.ones((batch_size, num_steps), dtype=torch.long), valid_lens),
    #                 (batch_size, num_steps, num_hiddens))
    #
    # #
    # num_hiddens, ffn_num_hiddens, num_heads = 24, 48, 8
    # encoder_blk = TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout=0.5)
    # block_idx = 0
    # decoder_blk = TransformerDecoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout=0.5, i=block_idx)
    # X = torch.ones((batch_size, num_steps, num_hiddens))
    # state = [encoder_blk(X, valid_lens), valid_lens, [None]]
    # d2l.check_shape(decoder_blk(X, state)[0], X.shape)

    #
    batch_size = 128
    data = d2l.MTFraEng(batch_size=batch_size)
    num_hiddens, num_blks, dropout = 256, 3, 0.2
    ffn_num_hiddens, num_heads = 64, 4

    encoder = TransformerEncoder(
        len(data.src_vocab), num_hiddens, ffn_num_hiddens, num_heads,
        num_blks, dropout)
    decoder = TransformerDecoder(
        len(data.tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads,
        num_blks, dropout)
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.0015)
    trainer = d2l.Trainer(max_epochs=3, gradient_clip_val=1, num_gpus=1)
    trainer.fit(model, data)

    #
    engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    print(f'data.num_steps: {data.num_steps}')  # data.num_steps: 9
    preds, _ = model.predict_step(
        data.build(engs, fras), d2l.try_gpu(), data.num_steps)
    for en, fr, p in zip(engs, fras, preds):
        translation = []
        for token in data.tgt_vocab.to_tokens(p):
            if token == '<eos>':
                break
            translation.append(token)
        print(f'{en} => {translation}, bleu,'
              f'{d2l.bleu(" ".join(translation), fr, k=2):.3f}')
