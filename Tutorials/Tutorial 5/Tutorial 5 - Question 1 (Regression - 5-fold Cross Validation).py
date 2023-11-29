import torch
from torch import nn
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, no_features, no_hidden, no_labels):
        super().__init__()
        self.mlp_stack = nn.Sequential(
            nn.Linear(no_features, no_hidden),
            nn.Sigmoid(),
            nn.Linear(no_hidden, no_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.mlp_stack(x)
        return logits
    
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
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
    
def train_exp(X, Y, no_folds, hidden_units, no_epochs, no_features, no_labels, patience, min_delta, lr):
    
    
    err = []
    for no_hidden in hidden_units:
        print(f"Hidden Neurons: {no_hidden}")
        cv = KFold(n_splits=no_folds, shuffle=True, random_state=1)
        fold = 0
        fold_err = []
        for train_idx, test_idx in cv.split(X, Y):
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]

            model = MLP(no_features, no_hidden, no_labels)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            early_stopper = EarlyStopper(patience, min_delta)
            for epoch in range(no_epochs):
                pred = model(torch.tensor(X_train, dtype=torch.float))
                loss = loss_fn(pred, torch.tensor(Y_train, dtype=torch.float))

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred = model(torch.tensor(X_test, dtype=torch.float))
                test_err = loss_fn(pred, torch.tensor(Y_test, dtype=torch.float))
                if early_stopper.early_stop(test_err):
                    break

            fold_err.append(test_err.item())
            print(f"fold: {fold:2d}\tTest error: {test_err}")
            fold += 1
        err.append(fold_err)
    print(f"err:\n{err}")
    cv_err = np.mean(np.array(err), axis=0)
    print(f"cv_err: \n{cv_err}")
    return cv_err

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

    exp_err = []
    print(f"\t\t\tHidden Units: {hidden_units}")
    for exp in range(no_exps):
        print(f"Experiment: {exp}")
        exp_err.append(
            train_exp(
                X, Y, 5, hidden_units, no_epochs, no_features, no_labels, patience, min_delta, lr
            )
        )
        print(f"Experiment: {exp}\tErrors: {np.array(exp_err[exp])}")
    print(f"exp_err: {exp_err}")
    mean_err = np.mean(np.array(exp_err), axis=0)
    print(f"Mean error: {mean_err}")
    print(f"hidden units: {hidden_units[np.argmin(mean_err)]}")

    # plot_data(X_train, Y_train, X_test, Y_test)

    # plot_mse(hidden_units, mean_err)
import torch
from torch import nn
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, no_features, no_hidden, no_labels):
        super().__init__()
        self.mlp_stack = nn.Sequential(
            nn.Linear(no_features, no_hidden),
            nn.Sigmoid(),
            nn.Linear(no_hidden, no_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.mlp_stack(x)
        return logits
    
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
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
    
def train_exp(X, Y, no_folds, hidden_units, no_epochs, no_features, no_labels, patience, min_delta, lr):
    
    
    err = []
    for no_hidden in hidden_units:
        print(f"Hidden Neurons: {no_hidden}")
        cv = KFold(n_splits=no_folds, shuffle=True, random_state=1)
        fold = 0
        fold_err = []
        for train_idx, test_idx in cv.split(X, Y):
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]

            model = MLP(no_features, no_hidden, no_labels)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            early_stopper = EarlyStopper(patience, min_delta)
            for epoch in range(no_epochs):
                pred = model(torch.tensor(X_train, dtype=torch.float))
                loss = loss_fn(pred, torch.tensor(Y_train, dtype=torch.float))

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred = model(torch.tensor(X_test, dtype=torch.float))
                test_err = loss_fn(pred, torch.tensor(Y_test, dtype=torch.float))
                if early_stopper.early_stop(test_err):
                    break

            fold_err.append(test_err.item())
            print(f"fold: {fold:2d}\tTest error: {test_err}")
            fold += 1
        err.append(fold_err)
    print(f"err:\n{err}")
    cv_err = np.mean(np.array(err), axis=0)
    print(f"cv_err: \n{cv_err}")
    return cv_err

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

    exp_err = []
    print(f"\t\t\tHidden Units: {hidden_units}")
    for exp in range(no_exps):
        print(f"Experiment: {exp}")
        exp_err.append(
            train_exp(
                X, Y, 5, hidden_units, no_epochs, no_features, no_labels, patience, min_delta, lr
            )
        )
        print(f"Experiment: {exp}\tErrors: {np.array(exp_err[exp])}")
    print(f"exp_err: {exp_err}")
    mean_err = np.mean(np.array(exp_err), axis=0)
    print(f"Mean error: {mean_err}")
    print(f"hidden units: {hidden_units[np.argmin(mean_err)]}")

    # plot_data(X_train, Y_train, X_test, Y_test)

    # plot_mse(hidden_units, mean_err)