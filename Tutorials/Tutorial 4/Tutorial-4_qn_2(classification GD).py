import torch
import numpy as np
import matplotlib.pylab as plt


class Two_Layer_DNN_Classifier:
    def __init__(self):
        init_V = [[1.01, 0.09, -0.39], [0.79, -0.45, -0.22], [0.28, 0.96, -0.07]]
        init_c = [0.0, 0.0, 0.0]

        init_W = [[-0.10, 0.97, 0.18], [-0.7, 0.38, 0.93]]
        init_b = [0.0, 0.0, 0.0]
        self.V = torch.tensor(init_V, dtype=torch.double)
        self.c = torch.tensor(init_c, dtype=torch.double)
        self.W = torch.tensor(init_W, dtype=torch.double)
        self.b = torch.tensor(init_b, dtype=torch.double)

    def __call__(self, x):
        z = torch.matmul(x, self.W) + self.b
        h = torch.sigmoid(z)
        u = torch.matmul(h, self.V) + self.c
        fu = torch.exp(u) / torch.sum(torch.exp(u), axis=1, keepdims=True)
        y = torch.argmax(fu, axis=1)
        return z, h, u, fu, y


def loss(k, y, fu):
    entropy = -torch.sum(torch.log(fu) * k)
    err = torch.sum(torch.not_equal(torch.argmax(k, dim=1), y))
    return entropy, err


def train(model, inputs, k, learning_rate):
    z, h, u, fu, y = model(inputs)
    # dy = y * (1 - y)
    grad_u = -(k - fu)
    grad_V = torch.matmul(torch.transpose(h, 0, 1), grad_u)
    grad_c = torch.sum(grad_u, axis=0)

    dh = h * (1 - h)
    grad_z = torch.matmul(grad_u, torch.transpose(model.V, 0, 1)) * dh
    grad_W = torch.matmul(torch.transpose(inputs, 0, 1), grad_z)
    grad_b = torch.sum(grad_z, axis=0)

    model.W -= learning_rate * grad_W
    model.b -= learning_rate * grad_b
    model.V -= learning_rate * grad_V
    model.c -= learning_rate * grad_c

    return grad_u, dh, grad_z


def plot_err(no_epochs, err, name):
    plt.figure()
    plt.plot(range(no_epochs), err)

    plt.xlabel("No Epochs")
    if name == "err":
        plt.ylabel("Classification error")
    else:
        plt.ylabel("Entropy")
    plt.show()




def plot_graph(X, Y, model, testX=None, testY=None):
    plt.figure()
    line = plt.subplot(1, 1, 1)
    plot_pred = line.plot(X[Y == 1, 0], X[Y == 1, 1], "b^", label="A")
    plot_original = line.plot(X[Y == 2, 0], X[Y == 2, 1], "ro", label="B")
    plot_original = line.plot(X[Y == 3, 0], X[Y == 3, 1], "gx", label="C")

    plt.plot(testX[testY == 1, 0], testX[testY == 1, 1], "cv", label="test A")
    plt.plot(testX[testY == 2, 0], testX[testY == 2, 1], "m.", label="test B")
    plt.plot(testX[testY == 3, 0], testX[testY == 3, 1], "yX", label="test C")
    ax = plt.gca()  # gca stands for 'get current axis'
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["bottom"].set_position(("data", 0))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("data", 0))
    plt.ylim(-4.5, 4.5)
    plt.xlim(-4.5, 4.5)
    plt.text(4.3, 0.1, r"$x_1$")
    plt.text(0.1, 4.3, r"$x_2$")
    plt.legend()
    # plt.figure()
    # plt.plot(X[Y == 1, 0], X[Y == 1, 1], "b^", label="A")
    # plt.plot(X[Y == 2, 0], X[Y == 2, 1], "ro", label="B")
    # plt.plot(X[Y == 3, 0], X[Y == 3, 1], "gx", label="C")
    # plt.xlabel("$x_1$")
    # plt.ylabel("$x_2$")
    # plt.title("Data points")
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": "{:12.8f}".format})

    lr = 0.1
    no_epochs = 500

    X = torch.tensor(
        np.array(
            [[1.0, 1.0], [0.0, 1.0], [3.0, 4.0], [2.0, 2.0], [2.0, -2.0], [-2.0, -3.0]]
        )
    )
    y = torch.tensor(np.array([1, 1, 2, 2, 3, 3]))
    K = torch.tensor(
        np.array(
            [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
        ).astype(float)
    )

    model = Two_Layer_DNN_Classifier()
    print(f"Initial W:\n{model.W}")
    print(f"Initial b:\n{model.b}\n")

    print(f"Initial V:\n{model.V}")
    print(f"Initial c:\n{model.c}\n")

    entropy, err = [], []
    for epoch in range(no_epochs):
        z_, h_, u_, fu_, y_ = model(X)
        entropy_, err_ = loss(K, y_, fu_)
        grad_u, dh, grad_z = train(model, X, K, lr)

        err.append(err_)
        entropy.append(entropy_)

        if epoch == 0:
            print(f"Epoch: {epoch}")
            print(f"Z:\n{z_}")
            print(f"H:\n{h_}")
            print(f"U:\n{u_}")
            print(f"Y:\n{y_}")

            print(f"Classification Error: {err_}")
            print(f"entropy: {entropy_}\n")
            print(f"grad J w.r.t. U:\n{grad_u}")
            print(f"grad J w.r.t. Z:\n{grad_z}")

            print(f"New W:\n{model.W}")
            print(f"New b:\n{model.b}\n")

            print(f"New V:\n{model.V}")
            print(f"New c:\n{model.c}\n")
            # break

    # plot_err(no_epochs, err, "err")
    # plot_err(no_epochs, entropy, "entropy")
    print(f"Converged W:\n{model.W}")
    print(f"Converged b:\n{model.b}\n")

    print(f"Converged V:\n{model.V}")
    print(f"Converged c:\n{model.c}\n")

    print(f"Entropy: {entropy[-1]}")
    print(f"Error: {err[-1]}")

    # Plot original inputs, then plot the line given by classification V, c
    # Plot x1 and x2
    # plot_graph(X, y, model)
    test_X = torch.tensor(np.array([[2.5, 1.5], [-1.5, 0.5]]))
    z_, h_, u_, fu_, y_ = model(test_X)
    y_ += 1
    
    print(f"Test input:\n{test_X}")
    print(f"fu:\n{fu_}")
    print(f"Test Class Label:\n{y_}")
    plot_graph(X, y, model, test_X, y_)