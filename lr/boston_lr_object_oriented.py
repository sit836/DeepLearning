import warnings

import pandas as pd
import torch
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

"""
    Reference: https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
"""

TARGET = 'Price'

torch.manual_seed(0)


def load_data():
    bos = load_boston()
    df = pd.DataFrame(bos.data)
    df.columns = bos.feature_names
    df[TARGET] = bos.target
    return df.drop(TARGET, axis=1).values, df[TARGET].values


def standardize_features(X_train_raw, X_test_raw):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    return X_train, X_test


def numpy2tensor(X_train_arr, X_test_arr, y_train_arr, y_test_arr):
    X_train = torch.tensor(X_train_arr, dtype=torch.float)
    X_test = torch.tensor(X_test_arr, dtype=torch.float)
    y_train = torch.tensor(y_train_arr, dtype=torch.float).reshape(-1, 1)
    y_test = torch.tensor(y_test_arr, dtype=torch.float).reshape(-1, 1)
    return X_train, X_test, y_train, y_test


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        x = self.layer(x)
        return x


batch_size_train = 10
num_epochs = 30
learning_rate = 0.01

print(torch.__version__)
print(torch.cuda.is_available())

X_arr, y_arr = load_data()

X_train_raw_arr, X_test_raw_arr, y_train_arr, y_test_arr = train_test_split(X_arr, y_arr, test_size=0.3,
                                                                            random_state=123)
print(f'X_train_raw_arr.shape, X_test_raw_arr.shape: {X_train_raw_arr.shape, X_test_raw_arr.shape}')

X_train_arr, X_test_arr = standardize_features(X_train_raw_arr, X_test_raw_arr)
X_train, X_test, y_train, y_test = numpy2tensor(X_train_arr, X_test_arr, y_train_arr, y_test_arr)
num_features = X_train.shape[1]

model = Model()

train_datasets = torch.utils.data.TensorDataset(X_train, y_train)
test_datasets = torch.utils.data.TensorDataset(X_test, y_test)
train_dl = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size_train, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_datasets, batch_size=1024, shuffle=False)

loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

training_loss = []
for epoch in tqdm(range(num_epochs)):
    for inputs_train, targets_train in train_dl:
        # print(inputs_train.shape)

        # clear the gradients
        optimizer.zero_grad()

        # __call__ method inherited from the nn.Module class turns model, an instance of the class, into callables: behave like a function
        pred_train = model(inputs_train)
        l = loss(pred_train, targets_train)

        # compute gradients w.r.t. parameters
        l.backward()

        # update model weights
        optimizer.step()

        # .item: Returns the value of the tensor
        training_loss.append(l.item())

# plt.plot(training_loss)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Training')
# plt.show()

pred_test_arr, targets_test_arr = None, None
test_loss = []
for inputs_test, targets_test in test_dl:
    pred_test = model(inputs_test)
    l = loss(pred_test, targets_test)
    test_loss.append(l.item())

    # detach the Tensor from the automatic differentiation graph and call the NumPy function
    pred_test_arr = pred_test.detach().numpy().flatten()
    targets_test_arr = targets_test.detach().numpy().flatten()

print(f'test_loss: {test_loss}')

plt.scatter(pred_test_arr, targets_test_arr)
plt.xlabel('Pred')
plt.ylabel('Data')
plt.title('Test')
plt.axline([0, 0], [1, 1], color='red', ls='--')
plt.show()
