from typing import Any
import torch
import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import make_blobs
import copy

np.set_printoptions(formatter={"float": "{:6.2f}".format})


def blob_maker(samples, features, std, centers):
    """
    Apparently, centers = number of classes, features = number of features
    Args:
        samples (_type_): Dataset size
        features (_type_): number of input features
        std (_type_): standard deviation of the generated dataset
        centers (_type_): number of classes

    Returns:
        _type_: The dataset and its class for each input
    """
    return make_blobs(
        n_samples=samples,
        n_features=features,
        cluster_std=std,
        centers=centers,
        random_state=1,
    )


class Logistic:
    def __init__(self):
        self.w = torch.tensor(
            np.random.rand(3, 1), dtype=torch.double, requires_grad=True
        )
        self.b = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
        print(f"w: \n{self.w}\nb: {self.b:5.2f}")

    def __call__(self, x):
        u = torch.matmul(torch.tensor(x), self.w) + self.b
        print(f"u:\n{u}")
        logits = torch.sigmoid(u)  # logits = f(u)
        print(f"fu:\n{logits}")
        return u, logits


def loss(targets, logits):
    entropy = torch.nn.BCELoss()
    print(f"logits:\n{logits}")
    print(f"targets:\n{targets}")
    entropy_ = entropy(logits, torch.tensor(targets, dtype=torch.double))
    class_err = torch.sum(torch.not_equal(logits > 0.5, torch.tensor(targets)))
    return entropy_, class_err


def train(model, inputs, targets, learning_rate):
    _, f_u = model(inputs)
    loss_, err_ = loss(targets, f_u)

    loss_.backward()
    with torch.no_grad():
        model.w -= learning_rate * model.w.grad
        model.b -= learning_rate * model.b.grad

    model.w.grad = None
    model.b.grad = None
    return loss_, err_


def plot_learning_curve(no_epochs, mse):
    """
    Plot learning curve of epoch vs mean square error.
    """
    plt.figure(1)
    plt.plot(range(no_epochs), mse)
    plt.xlabel("epochs")
    plt.ylabel("mse")
    plt.savefig(
        "./graphs/Tutorial 2 - Question 3 (Classification & sklearn) (epoch vs entropy loss).png"
    )
    plt.show()


def hyperplane(w, b):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)

    X, Y = np.meshgrid(x, y)
    Z = (
        w.detach().numpy()[0] * X + w.detach().numpy()[1] * Y + b.detach().numpy()
    ) / w.detach().numpy()[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, color="b", alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(
        "./graphs/Tutorial 2 - Question 3 (Classification & sklearn) (hyperplane).png"
    )
    plt.show()


def plot(X, Y, predictions):
    # fig = plt.figure(3)
    ax = plt.axes(projection="3d")
    ax.scatter(X[:, 0], X[:, 1], Y, color="blue", marker="x", label="targets")
    ax.scatter(
        X[:, 0],
        X[:, 1],
        np.array(predictions),
        color="red",
        marker=".",
        label="predicted",
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.legend()
    plt.savefig(
        "./graphs/Tutorial 2 - Question 3 (Classification & sklearn) (d vs f(u)).png"
    )
    # plt.show()

def plot2(X, Y, predictions):
    # fig = plt.figure(3)
    ax = plt.axes(projection="3d")
    ax.scatter(X[:, 1], X[:, 0], Y, color="blue", marker="x", label="targets")
    ax.scatter(
        X[:, 1],
        X[:, 0],
        np.array(predictions),
        color="red",
        marker=".",
        label="predicted",
    )
    ax.set_xlabel("$x_2$")
    ax.set_ylabel("$x_1$")
    ax.set_zlabel("$x_3$")
    ax.legend()
    # plt.savefig(
    #     "./graphs/Tutorial 2 - Question 3 (Classification & sklearn) (d vs f(u)).png"
    # )
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": "{:12.8f}".format})
    no_epochs = 30000
    lr = 0.01

    SEED = 10
    np.random.seed(SEED)

    X, Y = blob_maker(samples=100, features=3, std=5.0, centers=2)
    model = Logistic()
    og_w = copy.deepcopy(model.w)
    og_b = copy.deepcopy(model.b)

    entropy, error, prediction = [], [], []
    Y = Y.reshape(len(Y), 1)
    for epoch in range(no_epochs):
        entropy_, err_ = train(model, X, Y, lr)
        if epoch == 0:
            print(f"Epoch: {epoch}")
            print(f"mse: {entropy_:5.2f}")
            print(f"w: \n{model.w.detach().numpy()}")
            print(f"b: {model.b.detach().numpy()}")
            print()
        # if epoch % 500 == 0:
        if epoch > (no_epochs - 10):
            print("Epoch %2d:  entropy: %2.5f, error: %d" % (epoch, entropy_, err_))
            # print(f"Model w: \n{model.w.numpy()}\nModel b: \n\t{model.b.numpy()}")

        entropy.append(entropy_.detach().numpy()), error.append(err_.detach().numpy())
        break

    print(f"Current w\t\tOriginal w:")
    for i in range(len(model.w)):
        print(f"{model.w.detach().numpy()[i]}\t\t{og_w.detach().numpy()[i]}")
    print(f"Current b\t\tOriginal b:")
    print(f"{model.b.detach().numpy():9.2f}\t\t{og_b.detach().numpy():5.2f}")

    _, f_u = model(X)
    for i in range(len(f_u)):
        prediction.append((f_u > 0.5).numpy()[i].astype(int))

    plot_learning_curve(no_epochs, entropy)
    # plot(X, Y, prediction)
    # plot2(X, Y, prediction)

    hyperplane(model.w, model.b)

from typing import Any
import torch
import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import make_blobs
import copy

np.set_printoptions(formatter={"float": "{:6.2f}".format})


def blob_maker(samples, features, std, centers):
    """
    Apparently, centers = number of classes, features = number of features
    Args:
        samples (_type_): Dataset size
        features (_type_): number of input features
        std (_type_): standard deviation of the generated dataset
        centers (_type_): number of classes

    Returns:
        _type_: The dataset and its class for each input
    """
    return make_blobs(
        n_samples=samples,
        n_features=features,
        cluster_std=std,
        centers=centers,
        random_state=1,
    )


class Logistic:
    def __init__(self):
        self.w = torch.tensor(
            np.random.rand(3, 1), dtype=torch.double, requires_grad=True
        )
        self.b = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
        print(f"w: \n{self.w}\nb: {self.b:5.2f}")

    def __call__(self, x):
        u = torch.matmul(torch.tensor(x), self.w) + self.b
        print(f"u:\n{u}")
        logits = torch.sigmoid(u)  # logits = f(u)
        print(f"fu:\n{logits}")
        return u, logits


def loss(targets, logits):
    entropy = torch.nn.BCELoss()
    print(f"logits:\n{logits}")
    print(f"targets:\n{targets}")
    entropy_ = entropy(logits, torch.tensor(targets, dtype=torch.double))
    class_err = torch.sum(torch.not_equal(logits > 0.5, torch.tensor(targets)))
    return entropy_, class_err


def train(model, inputs, targets, learning_rate):
    _, f_u = model(inputs)
    loss_, err_ = loss(targets, f_u)

    loss_.backward()
    with torch.no_grad():
        model.w -= learning_rate * model.w.grad
        model.b -= learning_rate * model.b.grad

    model.w.grad = None
    model.b.grad = None
    return loss_, err_


def plot_learning_curve(no_epochs, mse):
    """
    Plot learning curve of epoch vs mean square error.
    """
    plt.figure(1)
    plt.plot(range(no_epochs), mse)
    plt.xlabel("epochs")
    plt.ylabel("mse")
    plt.savefig(
        "./graphs/Tutorial 2 - Question 3 (Classification & sklearn) (epoch vs entropy loss).png"
    )
    plt.show()


def hyperplane(w, b):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)

    X, Y = np.meshgrid(x, y)
    Z = (
        w.detach().numpy()[0] * X + w.detach().numpy()[1] * Y + b.detach().numpy()
    ) / w.detach().numpy()[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, color="b", alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(
        "./graphs/Tutorial 2 - Question 3 (Classification & sklearn) (hyperplane).png"
    )
    plt.show()


def plot(X, Y, predictions):
    # fig = plt.figure(3)
    ax = plt.axes(projection="3d")
    ax.scatter(X[:, 0], X[:, 1], Y, color="blue", marker="x", label="targets")
    ax.scatter(
        X[:, 0],
        X[:, 1],
        np.array(predictions),
        color="red",
        marker=".",
        label="predicted",
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.legend()
    plt.savefig(
        "./graphs/Tutorial 2 - Question 3 (Classification & sklearn) (d vs f(u)).png"
    )
    # plt.show()

def plot2(X, Y, predictions):
    # fig = plt.figure(3)
    ax = plt.axes(projection="3d")
    ax.scatter(X[:, 1], X[:, 0], Y, color="blue", marker="x", label="targets")
    ax.scatter(
        X[:, 1],
        X[:, 0],
        np.array(predictions),
        color="red",
        marker=".",
        label="predicted",
    )
    ax.set_xlabel("$x_2$")
    ax.set_ylabel("$x_1$")
    ax.set_zlabel("$x_3$")
    ax.legend()
    # plt.savefig(
    #     "./graphs/Tutorial 2 - Question 3 (Classification & sklearn) (d vs f(u)).png"
    # )
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": "{:12.8f}".format})
    no_epochs = 30000
    lr = 0.01

    SEED = 10
    np.random.seed(SEED)

    X, Y = blob_maker(samples=100, features=3, std=5.0, centers=2)
    model = Logistic()
    og_w = copy.deepcopy(model.w)
    og_b = copy.deepcopy(model.b)

    entropy, error, prediction = [], [], []
    Y = Y.reshape(len(Y), 1)
    for epoch in range(no_epochs):
        entropy_, err_ = train(model, X, Y, lr)
        if epoch == 0:
            print(f"Epoch: {epoch}")
            print(f"mse: {entropy_:5.2f}")
            print(f"w: \n{model.w.detach().numpy()}")
            print(f"b: {model.b.detach().numpy()}")
            print()
        # if epoch % 500 == 0:
        if epoch > (no_epochs - 10):
            print("Epoch %2d:  entropy: %2.5f, error: %d" % (epoch, entropy_, err_))
            # print(f"Model w: \n{model.w.numpy()}\nModel b: \n\t{model.b.numpy()}")

        entropy.append(entropy_.detach().numpy()), error.append(err_.detach().numpy())
        break

    print(f"Current w\t\tOriginal w:")
    for i in range(len(model.w)):
        print(f"{model.w.detach().numpy()[i]}\t\t{og_w.detach().numpy()[i]}")
    print(f"Current b\t\tOriginal b:")
    print(f"{model.b.detach().numpy():9.2f}\t\t{og_b.detach().numpy():5.2f}")

    _, f_u = model(X)
    for i in range(len(f_u)):
        prediction.append((f_u > 0.5).numpy()[i].astype(int))

    plot_learning_curve(no_epochs, entropy)
    # plot(X, Y, prediction)
    # plot2(X, Y, prediction)

    hyperplane(model.w, model.b)
