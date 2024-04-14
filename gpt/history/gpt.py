import os
import time
from collections import defaultdict

import numpy as np
import requests
import tiktoken
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from model import GPT

"""
    Reference: https://ai.plainenglish.io/creating-and-exploring-gpt-from-scratch-ffe84ac415a9
"""

# download the tiny shakespeare dataset
data_dir = os.path.join('../data', 'tinyshakespeare')
input_file_path = os.path.join(data_dir, 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    os.makedirs(data_dir)
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))


class GPTConfig:
    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        for key, value in kwargs.items():
            setattr(self, key, value)


class CustomConfig(GPTConfig):
    n_layer = 8
    n_head = 8
    n_embd = 256
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    dropout = 0.1
    compile = False
    device = 'cuda'
    num_workers = 0
    max_iters = 20000
    batch_size = 8
    block_size = 64
    learning_rate = 6e-4
    betas = (0.9, 0.95)
    weight_decay = 1e-1
    grad_norm_clip = 1.0


vocab_size = len(train_ids)
config = CustomConfig(vocab_size=vocab_size)

#
# read data from .bin
data_dir = os.path.join('../data', 'tinyshakespeare')
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')


class ShakespeareDataset(Dataset):
    def __init__(self, split, block_size=128, device_type='cuda'):
        assert split in {'train', 'test'}
        self.split = split
        self.block_size = block_size
        self.device_type = device_type
        self.data = train_data if split == 'train' else val_data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx: idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1: idx + 1 + self.block_size].astype(np.int64))

        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to('cuda', non_blocking=True), y.pin_memory().to('cuda', non_blocking=True)
        else:
            x, y = x.to('cpu'), y.to('cpu')
        return x, y


# create dataset and dataloader
train_dataset = ShakespeareDataset('train', config.block_size, config.device)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=False)
test_dataset = ShakespeareDataset('test', config.block_size, config.device)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=False)


class Trainer:

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.device = config.device
        self.model = self.model.to(self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            # pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break


model = GPT(config).to(config.device)
if config.compile:
    model = torch.compile(model)
trainer = Trainer(config, model, train_dataset)


def batch_end_callback(trainer):
    if trainer.iter_num % 500 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")


trainer.set_callback('on_batch_end', batch_end_callback)
trainer.run()

text = 'Lord:\nRise! My people, conquer the north!'
sample_ids = torch.Tensor(enc.encode_ordinary(text)).long()
sample_ids = torch.unsqueeze(sample_ids, 0).to(config.device)
result = model.generate(sample_ids, max_new_tokens=200, temperature=1, do_sample=False, top_k=None)
print(enc.decode(result.detach().cpu().tolist()[0]))
