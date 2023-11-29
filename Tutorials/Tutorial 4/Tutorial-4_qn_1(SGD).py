import torch
import numpy as np
import matplotlib.pylab as plt


class Two_Layer_DNN:
    def __init__(self):
        init_V = [[1.0, 1.0], [0.0, -2.0]]
        init_c = [-2.0, 3.0]

        init_W = [[1.0, 2.0], [-2.0, 0.0]]
        init_b = [3.0, -1.0]
        self.V = torch.tensor(init_V, dtype=torch.double)
        self.c = torch.tensor(init_c, dtype=torch.double)
        self.W = torch.tensor(init_W, dtype=torch.double)
        self.b = torch.tensor(init_b, dtype=torch.double)

    def __call__(self, x):
        z = torch.inner(torch.transpose(self.W, 0, 1), x) + self.b
        h = torch.sigmoid(z)
        u = torch.inner(torch.transpose(self.V, 0, 1), h) + self.c
        y = torch.sigmoid(u)
        return z, h, u, y


def loss(targets, outputs):
    # print(f"\ntargets:\n{targets}")
    # print(f"outputs:\n{outputs}")
    # print(f"Sum Square Error:\n{torch.square(targets - outputs)}\n")
    return torch.mean(torch.square(targets - outputs))


def train(model, inputs, d, learning_rate):
    z, h, u, y = model(inputs)
    dy = y * (1 - y)
    grad_u = -(d - y) * dy
    # grad_V = torch.matmul(h, torch.transpose(grad_u, 0, 1))
    grad_V = torch.outer(h, grad_u)
    grad_c = grad_u

    dh = h * (1 - h)
    grad_z = torch.inner(model.V, grad_u) * dh
    grad_W = torch.outer(inputs, grad_z)
    grad_b = grad_z

    model.W -= learning_rate * grad_W
    model.b -= learning_rate * grad_b
    model.V -= learning_rate * grad_V
    model.c -= learning_rate * grad_c

    return dy, grad_u, dh, grad_z


def plot_err(no_epochs, err):
    plt.figure()
    plt.plot(range(no_epochs), err)
    plt.xlabel("No Epochs")
    plt.ylabel("Mean Square Error")
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": "{:12.8f}".format})
    SEED = 10
    np.random.seed(SEED)

    lr = 0.05
    no_epochs = 5000
    X = torch.tensor(np.array([[1.0, 3.0], [-2.0, -2.0]]))
    Y = torch.tensor(np.array([[0.0, 1.0], [1.0, 0.0]]))

    model = Two_Layer_DNN()
    print(f"Initial W:\n{model.W}")
    print(f"Initial b:\n{model.b}\n")

    print(f"Initial V:\n{model.V}")
    print(f"Initial c:\n{model.c}\n")

    err = []
    idx = np.arange(2)

    for epoch in range(no_epochs):
        np.random.shuffle(idx)
        X, Y = X[idx], Y[idx]
        cost = []
        for p in [0, 1]:
            z_, h_, u_, y_ = model(X[p])
            cost_ = loss(Y[p], y_)
            dy, grad_u, dh, grad_z = train(model, X[p], Y[p], lr)

            cost.append(cost_)

            if epoch == 0:
                print(f"Epoch: {epoch}\nPattern:")
                print(f"X: \n{X[p]}")
                print(f"Y: \n{Y[p]}")
                print(f"Z:\n{z_}")
                print(f"H:\n{h_}")
                print(f"U:\n{u_}")
                print(f"Y:\n{y_}")

                print(f"cost: {cost}")
                print(f"grad J w.r.t. U:\n{grad_u}")
                print(f"grad J w.r.t. Z:\n{grad_z}")

                print(f"New W:\n{model.W}")
                print(f"New b:\n{model.b}\n")

                print(f"New V:\n{model.V}")
                print(f"New c:\n{model.c}\n")
                # break

        err.append(np.mean(cost))
    print(f"Final Error: {err[-1]}")
    print(f"err: {len(err)}")
    plot_err(no_epochs, err)
    print(f"Converged W:\n{model.W}")
    print(f"Converged b:\n{model.b}\n")

    print(f"Converged V:\n{model.V}")
    print(f"Converged c:\n{model.c}\n")

import torch
import numpy as np
import matplotlib.pylab as plt


class Two_Layer_DNN:
    def __init__(self):
        init_V = [[1.0, 1.0], [0.0, -2.0]]
        init_c = [-2.0, 3.0]

        init_W = [[1.0, 2.0], [-2.0, 0.0]]
        init_b = [3.0, -1.0]
        self.V = torch.tensor(init_V, dtype=torch.double)
        self.c = torch.tensor(init_c, dtype=torch.double)
        self.W = torch.tensor(init_W, dtype=torch.double)
        self.b = torch.tensor(init_b, dtype=torch.double)

    def __call__(self, x):
        z = torch.inner(torch.transpose(self.W, 0, 1), x) + self.b
        h = torch.sigmoid(z)
        u = torch.inner(torch.transpose(self.V, 0, 1), h) + self.c
        y = torch.sigmoid(u)
        return z, h, u, y


def loss(targets, outputs):
    # print(f"\ntargets:\n{targets}")
    # print(f"outputs:\n{outputs}")
    # print(f"Sum Square Error:\n{torch.square(targets - outputs)}\n")
    return torch.mean(torch.square(targets - outputs))


def train(model, inputs, d, learning_rate):
    z, h, u, y = model(inputs)
    dy = y * (1 - y)
    grad_u = -(d - y) * dy
    # grad_V = torch.matmul(h, torch.transpose(grad_u, 0, 1))
    grad_V = torch.outer(h, grad_u)
    grad_c = grad_u

    dh = h * (1 - h)
    grad_z = torch.inner(model.V, grad_u) * dh
    grad_W = torch.outer(inputs, grad_z)
    grad_b = grad_z

    model.W -= learning_rate * grad_W
    model.b -= learning_rate * grad_b
    model.V -= learning_rate * grad_V
    model.c -= learning_rate * grad_c

    return dy, grad_u, dh, grad_z


def plot_err(no_epochs, err):
    plt.figure()
    plt.plot(range(no_epochs), err)
    plt.xlabel("No Epochs")
    plt.ylabel("Mean Square Error")
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": "{:12.8f}".format})
    SEED = 10
    np.random.seed(SEED)

    lr = 0.05
    no_epochs = 5000
    X = torch.tensor(np.array([[1.0, 3.0], [-2.0, -2.0]]))
    Y = torch.tensor(np.array([[0.0, 1.0], [1.0, 0.0]]))

    model = Two_Layer_DNN()
    print(f"Initial W:\n{model.W}")
    print(f"Initial b:\n{model.b}\n")

    print(f"Initial V:\n{model.V}")
    print(f"Initial c:\n{model.c}\n")

    err = []
    idx = np.arange(2)

    for epoch in range(no_epochs):
        np.random.shuffle(idx)
        X, Y = X[idx], Y[idx]
        cost = []
        for p in [0, 1]:
            z_, h_, u_, y_ = model(X[p])
            cost_ = loss(Y[p], y_)
            dy, grad_u, dh, grad_z = train(model, X[p], Y[p], lr)

            cost.append(cost_)

            if epoch == 0:
                print(f"Epoch: {epoch}\nPattern:")
                print(f"X: \n{X[p]}")
                print(f"Y: \n{Y[p]}")
                print(f"Z:\n{z_}")
                print(f"H:\n{h_}")
                print(f"U:\n{u_}")
                print(f"Y:\n{y_}")

                print(f"cost: {cost}")
                print(f"grad J w.r.t. U:\n{grad_u}")
                print(f"grad J w.r.t. Z:\n{grad_z}")

                print(f"New W:\n{model.W}")
                print(f"New b:\n{model.b}\n")

                print(f"New V:\n{model.V}")
                print(f"New c:\n{model.c}\n")
                # break

        err.append(np.mean(cost))
    print(f"Final Error: {err[-1]}")
    print(f"err: {len(err)}")
    plot_err(no_epochs, err)
    print(f"Converged W:\n{model.W}")
    print(f"Converged b:\n{model.b}\n")

    print(f"Converged V:\n{model.V}")
    print(f"Converged c:\n{model.c}\n")
