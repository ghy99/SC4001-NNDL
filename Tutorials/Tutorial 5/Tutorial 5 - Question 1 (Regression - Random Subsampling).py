import torch
from torch import nn
import numpy as np
import matplotlib.pylab as plt


class MLP(nn.Module):
    def __init__(self, no_features, no_hidden, no_labels):
        super().__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(no_features, no_hidden),
            nn.Sigmoid(),
            nn.Linear(no_hidden, no_labels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.relu_stack(x)
        return logits


class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_exp(X_train, Y_train, X_test, Y_test, hidden_units, patience, min_delta):
    test_loss_ = []
    for no_hidden in hidden_units:
        model = MLP(no_features, no_hidden, no_labels)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
        test_loss = None
        for epoch in range(no_epochs):
            pred = model(torch.tensor(X_train, dtype=torch.float))
            train_loss = loss_fn(pred, torch.tensor(Y_train, dtype=torch.float))

            # Backpropagation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            pred = model(torch.tensor(X_test, dtype=torch.float))
            test_loss = loss_fn(pred, torch.tensor(Y_test, dtype=torch.float))

            if early_stopper.early_stop(test_loss):
                break
        # Track test loss only to see final performance
        test_loss_.append(test_loss.item())
    return test_loss_


def plot_meshgrid(X, Y):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    X1 = np.arange(-1, 1, 0.05)
    X2 = np.arange(-1, 1, 0.05)
    X1, X2 = np.meshgrid(X1, X2)
    Z = np.sin(np.pi * X1) * np.cos(2 * np.pi * X2)
    ax.plot_surface(X1, X2, Z)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$y$")
    ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_zticks([-1.0, -0.5, 0, 0.5, 1.0])
    plt.show()


def plot_data(X_train, Y_train, X_test, Y_test):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(X_train[:, 0], X_train[:, 1], Y_train[:, 0], "b.", label="train")
    ax.scatter(X_test[:, 0], X_test[:, 1], Y_test[:, 0], "rx", label="test")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$y$")
    ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_zticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.legend()
    plt.show()


def plot_mse(hidden_units, mean_err):
    plt.figure()
    plt.plot(hidden_units, mean_err, marker="x", linestyle="None")
    plt.xticks(hidden_units)
    plt.xlabel("Number of hidden units")
    plt.ylabel("Mean Error")
    plt.show()


if __name__ == "__main__":
    seed = 10
    np.random.seed(seed)
    no_features = 2
    hidden_units = [2, 4, 6, 8, 10]
    no_labels = 1
    no_exps = 10

    lr = 0.05
    no_epochs = 5000
    patience = 10
    min_delta = 0

    # Generating Training Data
    X = np.zeros((10 * 10, no_features))
    no_data = 0
    for i in np.arange(-1.0, 1.001, 2.0 / 9.0):
        for j in np.arange(-1.0, 1.001, 2.0 / 9.0):
            X[no_data] = [i, j]
            no_data += 1

    Y = np.zeros((no_data, 1))
    Y[:, 0] = np.sin(np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])

    err = []
    print(f"\t\t\tHidden Units: {hidden_units}")
    for exp in range(no_exps):
        idx = np.arange(no_data)
        np.random.shuffle(idx)
        X, Y = X[idx], Y[idx]
        X_train, Y_train, X_test, Y_test = X[:70], Y[:70], X[70:], Y[70:]
        err.append(
            train_exp(
                X_train, Y_train, X_test, Y_test, hidden_units, patience, min_delta
            )
        )
        print(f"Experiment: {exp}\tErrors: {np.array(err[exp])}")

    mean_err = np.mean(np.array(err), axis=0)
    print(f"Mean error: {mean_err}")
    print(f"hidden units: {hidden_units[np.argmin(mean_err)]}")

    plot_data(X_train, Y_train, X_test, Y_test)

    plot_mse(hidden_units, mean_err)
import torch
from torch import nn
import numpy as np
import matplotlib.pylab as plt


class MLP(nn.Module):
    def __init__(self, no_features, no_hidden, no_labels):
        super().__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(no_features, no_hidden),
            nn.Sigmoid(),
            nn.Linear(no_hidden, no_labels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.relu_stack(x)
        return logits


class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_exp(X_train, Y_train, X_test, Y_test, hidden_units, patience, min_delta):
    test_loss_ = []
    for no_hidden in hidden_units:
        model = MLP(no_features, no_hidden, no_labels)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
        test_loss = None
        for epoch in range(no_epochs):
            pred = model(torch.tensor(X_train, dtype=torch.float))
            train_loss = loss_fn(pred, torch.tensor(Y_train, dtype=torch.float))

            # Backpropagation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            pred = model(torch.tensor(X_test, dtype=torch.float))
            test_loss = loss_fn(pred, torch.tensor(Y_test, dtype=torch.float))

            if early_stopper.early_stop(test_loss):
                break
        # Track test loss only to see final performance
        test_loss_.append(test_loss.item())
    return test_loss_


def plot_meshgrid(X, Y):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    X1 = np.arange(-1, 1, 0.05)
    X2 = np.arange(-1, 1, 0.05)
    X1, X2 = np.meshgrid(X1, X2)
    Z = np.sin(np.pi * X1) * np.cos(2 * np.pi * X2)
    ax.plot_surface(X1, X2, Z)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$y$")
    ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_zticks([-1.0, -0.5, 0, 0.5, 1.0])
    plt.show()


def plot_data(X_train, Y_train, X_test, Y_test):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(X_train[:, 0], X_train[:, 1], Y_train[:, 0], "b.", label="train")
    ax.scatter(X_test[:, 0], X_test[:, 1], Y_test[:, 0], "rx", label="test")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$y$")
    ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_zticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.legend()
    plt.show()


def plot_mse(hidden_units, mean_err):
    plt.figure()
    plt.plot(hidden_units, mean_err, marker="x", linestyle="None")
    plt.xticks(hidden_units)
    plt.xlabel("Number of hidden units")
    plt.ylabel("Mean Error")
    plt.show()


if __name__ == "__main__":
    seed = 10
    np.random.seed(seed)
    no_features = 2
    hidden_units = [2, 4, 6, 8, 10]
    no_labels = 1
    no_exps = 10

    lr = 0.05
    no_epochs = 5000
    patience = 10
    min_delta = 0

    # Generating Training Data
    X = np.zeros((10 * 10, no_features))
    no_data = 0
    for i in np.arange(-1.0, 1.001, 2.0 / 9.0):
        for j in np.arange(-1.0, 1.001, 2.0 / 9.0):
            X[no_data] = [i, j]
            no_data += 1

    Y = np.zeros((no_data, 1))
    Y[:, 0] = np.sin(np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])

    err = []
    print(f"\t\t\tHidden Units: {hidden_units}")
    for exp in range(no_exps):
        idx = np.arange(no_data)
        np.random.shuffle(idx)
        X, Y = X[idx], Y[idx]
        X_train, Y_train, X_test, Y_test = X[:70], Y[:70], X[70:], Y[70:]
        err.append(
            train_exp(
                X_train, Y_train, X_test, Y_test, hidden_units, patience, min_delta
            )
        )
        print(f"Experiment: {exp}\tErrors: {np.array(err[exp])}")

    mean_err = np.mean(np.array(err), axis=0)
    print(f"Mean error: {mean_err}")
    print(f"hidden units: {hidden_units[np.argmin(mean_err)]}")

    plot_data(X_train, Y_train, X_test, Y_test)

    plot_mse(hidden_units, mean_err)