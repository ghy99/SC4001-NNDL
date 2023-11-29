import torch
import numpy as np
import matplotlib.pylab as plt

no_epochs = 200
lr = 0.01

X = np.array(
    [
        [0.09, -0.44, -0.15],
        [0.69, -0.99, -0.76],
        [0.34, 0.65, -0.73],
        [0.15, 0.78, -0.58],
        [-0.63, -0.78, -0.56],
        [0.96, 0.62, -0.66],
        [0.63, -0.45, -0.14],
        [0.88, 0.64, -0.33],
    ]
)
Y = np.array([-2.57, -2.97, 0.96, 1.04, -3.21, 1.05, -2.39, 0.66])

print("X: {}".format(X))
print("Y: {}".format(Y))


class Linear:
    def __init__(self):
        """
        Initialise weights and biases
        """
        self.w = torch.tensor(np.array([0.77, 0.02, 0.63]), dtype=torch.double)
        self.b = torch.tensor(0.0, dtype=torch.double)

    def __call__(self, x):
        """
        Calculate and return output of a linear neuron:
        y = Xw + b(1p)
        """
        return torch.matmul(torch.tensor(x), self.w) + self.b


def loss(predicted_y, target_y):
    """
    squared error as the loss function J
    """
    # Why is it predicted - target and not target - predicted? isn't the formula (d-y)^2?
    # Summation of (y - d)^2
    return torch.mean(torch.square(predicted_y - torch.tensor(target_y)))


def train_step(model, x, d, learning_rate):
    y = model(x)
    loss_ = loss(y, d)
    # Matrix Multiplication of X and (vector d and vector y)
    # w <- w + a(X^T)(d - y)
    # b <- b + a(1p^T)(d - y)
    grad_w = -torch.matmul(
        torch.transpose(torch.tensor(x), 0, 1), (torch.tensor(d) - y)
    )
    grad_b = -torch.sum((torch.tensor(d) - y))
    model.w -= learning_rate * grad_w
    model.b -= learning_rate * grad_b


model = Linear()
print("w: {}, b: {}".format(model.w.numpy(), model.b.numpy()))
print(f"learning factor: {lr}")

print(f"\n\nTraining starts here\n\n")
"""Keep an index for training"""
idx = np.arange(len(X))

"""Store square error J"""
mse = []
for epoch in range(no_epochs):
    y_ = model(X)
    loss_ = loss(y_, Y)

    train_step(model, X, Y, learning_rate=lr)

    mse.append(loss_)

    if epoch == 0:
        print("epoch (iteration): {}".format(epoch + 1))
        print(f"\toutput of linear neuron (y): {y_}")
        print(f"\tmean square error (m.s.e.): {loss_:.4f}")
        print(
            f"\tweight (w): {np.around(model.w.numpy(), 4)},\tbias (b): {model.b.numpy():.4f}"
        )
        print()

    if epoch % 20 == 0:
        print(f"epoch: {epoch}")
        print("\titer: %3d, mse: %1.4f" % (epoch, mse[epoch]))

        print(
            f"\tweight (w): {np.around(model.w.numpy(), 4)},\tbias (b): {model.b.numpy():.4f}"
        )

# print learning curve
plt.figure(1)
plt.plot(range(no_epochs), mse)
plt.xlabel("epochs")
plt.ylabel("mse")
plt.show()

pred = []
for p in np.arange(len(X)):
    pred.append(model(X[p]).numpy())

print(f"Predicted y: \t{pred}")
print(f"Target y: \t{Y}")

import torch
import numpy as np
import matplotlib.pylab as plt

no_epochs = 200
lr = 0.01

X = np.array(
    [
        [0.09, -0.44, -0.15],
        [0.69, -0.99, -0.76],
        [0.34, 0.65, -0.73],
        [0.15, 0.78, -0.58],
        [-0.63, -0.78, -0.56],
        [0.96, 0.62, -0.66],
        [0.63, -0.45, -0.14],
        [0.88, 0.64, -0.33],
    ]
)
Y = np.array([-2.57, -2.97, 0.96, 1.04, -3.21, 1.05, -2.39, 0.66])

print("X: {}".format(X))
print("Y: {}".format(Y))


class Linear:
    def __init__(self):
        """
        Initialise weights and biases
        """
        self.w = torch.tensor(np.array([0.77, 0.02, 0.63]), dtype=torch.double)
        self.b = torch.tensor(0.0, dtype=torch.double)

    def __call__(self, x):
        """
        Calculate and return output of a linear neuron:
        y = Xw + b(1p)
        """
        return torch.matmul(torch.tensor(x), self.w) + self.b


def loss(predicted_y, target_y):
    """
    squared error as the loss function J
    """
    # Why is it predicted - target and not target - predicted? isn't the formula (d-y)^2?
    # Summation of (y - d)^2
    return torch.mean(torch.square(predicted_y - torch.tensor(target_y)))


def train_step(model, x, d, learning_rate):
    y = model(x)
    loss_ = loss(y, d)
    # Matrix Multiplication of X and (vector d and vector y)
    # w <- w + a(X^T)(d - y)
    # b <- b + a(1p^T)(d - y)
    grad_w = -torch.matmul(
        torch.transpose(torch.tensor(x), 0, 1), (torch.tensor(d) - y)
    )
    grad_b = -torch.sum((torch.tensor(d) - y))
    model.w -= learning_rate * grad_w
    model.b -= learning_rate * grad_b


model = Linear()
print("w: {}, b: {}".format(model.w.numpy(), model.b.numpy()))
print(f"learning factor: {lr}")

print(f"\n\nTraining starts here\n\n")
"""Keep an index for training"""
idx = np.arange(len(X))

"""Store square error J"""
mse = []
for epoch in range(no_epochs):
    y_ = model(X)
    loss_ = loss(y_, Y)

    train_step(model, X, Y, learning_rate=lr)

    mse.append(loss_)

    if epoch == 0:
        print("epoch (iteration): {}".format(epoch + 1))
        print(f"\toutput of linear neuron (y): {y_}")
        print(f"\tmean square error (m.s.e.): {loss_:.4f}")
        print(
            f"\tweight (w): {np.around(model.w.numpy(), 4)},\tbias (b): {model.b.numpy():.4f}"
        )
        print()

    if epoch % 20 == 0:
        print(f"epoch: {epoch}")
        print("\titer: %3d, mse: %1.4f" % (epoch, mse[epoch]))

        print(
            f"\tweight (w): {np.around(model.w.numpy(), 4)},\tbias (b): {model.b.numpy():.4f}"
        )

# print learning curve
plt.figure(1)
plt.plot(range(no_epochs), mse)
plt.xlabel("epochs")
plt.ylabel("mse")
plt.show()

pred = []
for p in np.arange(len(X)):
    pred.append(model(X[p]).numpy())

print(f"Predicted y: \t{pred}")
print(f"Target y: \t{Y}")
