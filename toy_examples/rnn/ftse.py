import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


"""
Given data going as far back as "lookback" timesteps and sampled every "step" timesteps,
can we predict the target in "delay" timesteps?
"""
IN_PATH = './data/'

lookback = 60
step = 1
delay = 5
batch_size = 128

TARGET_NAME = 'Open'

df = pd.read_csv(os.path.join(IN_PATH, 'INDEX_UK_FTSE UK_UKX.csv'))

df.replace(',', '', inplace=True)

print(f'df.shape: {df.shape}')
print(f'df.dtypes: {df.dtypes}')

plt.plot(df[TARGET_NAME])
plt.show()
