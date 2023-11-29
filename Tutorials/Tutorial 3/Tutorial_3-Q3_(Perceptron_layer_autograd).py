import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
import torch
from torch import nn

from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split

# preprocess input and output data
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score


np.set_printoptions(formatter={"float": "{:12.8f}".format})

SEED = 100
np.random.seed(SEED)
torch.manual_seed(SEED)

X, y = load_linnerud(return_X_y=True)

# print(f"X:\n{X}")
# print(f"y:\n{y}")

# df_X = pd.DataFrame(X, columns=["Chins", "Situps", "Jumps"])
# df_y = pd.DataFrame(y, columns=["Weight", "Waist", "Pulse"])

# df = pd.merge(df_y, df_X, left_index=True, right_index=True)
# print(df.describe())

# corrmat = df.corr()
# f, ax = plt.subplots(figsize=(8, 6))
# sns.heatmap(corrmat, vmax=.8, square=True)
# plt.show()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

print(f"x_train shape:\t {x_train.shape}")
print(f"x_test shape:\t {x_test.shape}")
print(f"y_train shape:\t {y_train.shape}")
print(f"y_test shape:\t {y_test.shape}")

X_scaler = preprocessing.StandardScaler().fit(x_train)
X_scaled = X_scaler.transform(x_train)

y_scaler = preprocessing.MinMaxScaler().fit(y_train)
y_scaled = y_scaler.transform(y_train)

print(f"X_scaler mean:\t\t {X_scaler.mean_}")
print(f"X_scaler variance:\t {X_scaler.var_}")

print(f"y_scaler scale:\t\t {y_scaler.scale_}")
print(f"y_scaler min:\t {y_scaler.min_}")


class Perceptron_Layer(nn.Module):
    def __init__(self, no_inputs, no_outputs):
        super().__init__()
        self.perceptron_layer = nn.Sequential(
            nn.Linear(no_inputs, no_outputs), nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.perceptron_layer(x)
        return logits


# def loss_fn(predicted, target):
#     """
#     Finding mean squared error, not sum of squared error.
#     """
#     return torch.mean(torch.square(predicted - torch.tensor(target)))


def train(model, inputs, outputs, learning_rate):
    y = model(inputs)
    dy = y * (1 - y)
    grad_u = -(torch.tensor(outputs) - y) * dy
    grad_w = torch.matmul(torch.transpose(torch.tensor(inputs), 0, 1), grad_u)
    grad_b = torch.sum(grad_u, dim=0)

    model.w -= learning_rate * grad_w
    model.b -= learning_rate * grad_b


# 3 inputs, 3 outputs as from the structure of the data
no_features = 3
no_labels = 3

model = Perceptron_Layer(no_features, no_labels)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

w = model.state_dict()["perceptron_layer.0.weight"].numpy()
b = model.state_dict()["perceptron_layer.0.bias"].numpy()
print(f"weight:\n{w}\nbias:\n{b}")

no_epochs = 50000
# lr = 0.001

cost = []
t = 0
# idx = np.arange(len(X_scaled))

for epoch in range(no_epochs):
    start = time.time()

    predicted_y = model(torch.tensor(X_scaled, dtype=torch.float))
    loss = loss_fn(predicted_y, torch.tensor(y_scaled, dtype=torch.float))

    optimizer.zero_grad()
    loss.backward()  # Differentiation of J
    optimizer.step()  # Parameter update
    cost.append(loss.item())
    t += time.time() - start

    if epoch % 1000 == 0:
        print(f"\nEpoch: {epoch}")
        print(f"Loss: {loss.item()}")
        w = model.state_dict()["perceptron_layer.0.weight"].numpy()
        b = model.state_dict()["perceptron_layer.0.bias"].numpy()
        print(f"weight:\n{w}\nbias:\n{b}")

print()


def plot_learning_curve(no_epochs, cost):
    plt.figure()
    plt.plot(range(no_epochs), cost)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Square Error")
    plt.title("Autograd function with learning rate = 0.001")
    plt.show()
    plt.clf()


plot_learning_curve(no_epochs, cost)

X_scaled = X_scaler.transform(x_test)
y_pred = model(torch.tensor(X_scaled, dtype=torch.float))
y_scaled = (y_pred - torch.tensor(y_scaler.min_)) / torch.tensor(y_scaler.scale_) # IDK what this does
# y_scaled = y_scaler.inverse_transform(y_pred.detach().numpy())
loss = loss_fn(y_scaled, torch.tensor(y_test, dtype=torch.float)).item()

y_hat = y_scaled.detach().numpy()
print(f"y_hat:\n{y_hat}")
print(f"y_test:\n{y_test}")

print(f"\nLoss: {loss}")

rms = mean_squared_error(y_scaled.detach().numpy(), y_test, squared=True, multioutput="raw_values")
print(f"Root Mean Squared Error: {rms}")

r2 = r2_score(y_scaled.detach().numpy(), y_test, multioutput="raw_values")
print(f"R^2 values: {r2}")
print(f"Time taken = {t}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
import torch
from torch import nn

from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split

# preprocess input and output data
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score


np.set_printoptions(formatter={"float": "{:12.8f}".format})

SEED = 100
np.random.seed(SEED)
torch.manual_seed(SEED)

X, y = load_linnerud(return_X_y=True)

# print(f"X:\n{X}")
# print(f"y:\n{y}")

# df_X = pd.DataFrame(X, columns=["Chins", "Situps", "Jumps"])
# df_y = pd.DataFrame(y, columns=["Weight", "Waist", "Pulse"])

# df = pd.merge(df_y, df_X, left_index=True, right_index=True)
# print(df.describe())

# corrmat = df.corr()
# f, ax = plt.subplots(figsize=(8, 6))
# sns.heatmap(corrmat, vmax=.8, square=True)
# plt.show()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

print(f"x_train shape:\t {x_train.shape}")
print(f"x_test shape:\t {x_test.shape}")
print(f"y_train shape:\t {y_train.shape}")
print(f"y_test shape:\t {y_test.shape}")

X_scaler = preprocessing.StandardScaler().fit(x_train)
X_scaled = X_scaler.transform(x_train)

y_scaler = preprocessing.MinMaxScaler().fit(y_train)
y_scaled = y_scaler.transform(y_train)

print(f"X_scaler mean:\t\t {X_scaler.mean_}")
print(f"X_scaler variance:\t {X_scaler.var_}")

print(f"y_scaler scale:\t\t {y_scaler.scale_}")
print(f"y_scaler min:\t {y_scaler.min_}")


class Perceptron_Layer(nn.Module):
    def __init__(self, no_inputs, no_outputs):
        super().__init__()
        self.perceptron_layer = nn.Sequential(
            nn.Linear(no_inputs, no_outputs), nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.perceptron_layer(x)
        return logits


# def loss_fn(predicted, target):
#     """
#     Finding mean squared error, not sum of squared error.
#     """
#     return torch.mean(torch.square(predicted - torch.tensor(target)))


def train(model, inputs, outputs, learning_rate):
    y = model(inputs)
    dy = y * (1 - y)
    grad_u = -(torch.tensor(outputs) - y) * dy
    grad_w = torch.matmul(torch.transpose(torch.tensor(inputs), 0, 1), grad_u)
    grad_b = torch.sum(grad_u, dim=0)

    model.w -= learning_rate * grad_w
    model.b -= learning_rate * grad_b


# 3 inputs, 3 outputs as from the structure of the data
no_features = 3
no_labels = 3

model = Perceptron_Layer(no_features, no_labels)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

w = model.state_dict()["perceptron_layer.0.weight"].numpy()
b = model.state_dict()["perceptron_layer.0.bias"].numpy()
print(f"weight:\n{w}\nbias:\n{b}")

no_epochs = 50000
# lr = 0.001

cost = []
t = 0
# idx = np.arange(len(X_scaled))

for epoch in range(no_epochs):
    start = time.time()

    predicted_y = model(torch.tensor(X_scaled, dtype=torch.float))
    loss = loss_fn(predicted_y, torch.tensor(y_scaled, dtype=torch.float))

    optimizer.zero_grad()
    loss.backward()  # Differentiation of J
    optimizer.step()  # Parameter update
    cost.append(loss.item())
    t += time.time() - start

    if epoch % 1000 == 0:
        print(f"\nEpoch: {epoch}")
        print(f"Loss: {loss.item()}")
        w = model.state_dict()["perceptron_layer.0.weight"].numpy()
        b = model.state_dict()["perceptron_layer.0.bias"].numpy()
        print(f"weight:\n{w}\nbias:\n{b}")

print()


def plot_learning_curve(no_epochs, cost):
    plt.figure()
    plt.plot(range(no_epochs), cost)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Square Error")
    plt.title("Autograd function with learning rate = 0.001")
    plt.show()
    plt.clf()


plot_learning_curve(no_epochs, cost)

X_scaled = X_scaler.transform(x_test)
y_pred = model(torch.tensor(X_scaled, dtype=torch.float))
y_scaled = (y_pred - torch.tensor(y_scaler.min_)) / torch.tensor(y_scaler.scale_) # IDK what this does
# y_scaled = y_scaler.inverse_transform(y_pred.detach().numpy())
loss = loss_fn(y_scaled, torch.tensor(y_test, dtype=torch.float)).item()

y_hat = y_scaled.detach().numpy()
print(f"y_hat:\n{y_hat}")
print(f"y_test:\n{y_test}")

print(f"\nLoss: {loss}")

rms = mean_squared_error(y_scaled.detach().numpy(), y_test, squared=True, multioutput="raw_values")
print(f"Root Mean Squared Error: {rms}")

r2 = r2_score(y_scaled.detach().numpy(), y_test, multioutput="raw_values")
print(f"R^2 values: {r2}")
print(f"Time taken = {t}")
