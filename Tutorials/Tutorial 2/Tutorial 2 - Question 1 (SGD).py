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
        self.w = torch.tensor(np.array([0.77, 0.02, 0.63]))
        self.b = torch.tensor(0.0)

    def __call__(self, x):
        """
        Calculate and return output of a linear neuron:
        y = wTx + b
        """
        return torch.inner(torch.tensor(x), self.w) + self.b


def loss(predicted_y, target_y):
    """
    squared error as the loss function J
    """
    # Why is it predicted - target and not target - predicted? isn't the formula (d-y)^2?
    return torch.square(predicted_y - torch.tensor(target_y))


def train_step(model, x, d, learning_rate):
    y = model(x)
    loss_ = loss(y, d)
    grad_w = -(d - y) * x
    grad_b = -(d - y)
    model.w -= learning_rate * grad_w
    model.b -= learning_rate * grad_b


model = Linear()
print("w: {}, b: {}".format(model.w.numpy(), model.b.numpy()))
print(f"learning factor: {lr}")

print(f"\n\nTraining starts here\n\n")
"""Keep an index for training"""
idx = np.arange(len(X))

"""Store square error J"""
err = []
for epoch in range(no_epochs):
    """SGD need to shuffle pattern for each epoch"""
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]

    err_ = []
    for pattern in np.arange(len(X)):
        y_ = model(X[pattern])
        loss_ = loss(y_, Y[pattern])

        train_step(model, X[pattern], Y[pattern], learning_rate=lr)
        err_.append(loss_)

        if epoch == 0:
            print("epoch (iteration): {}".format(epoch + 1))
            print("\tpattern: {}".format(pattern + 1))
            print(
                "\tinput (x): {},\ttarget label (d): {}".format(X[pattern], Y[pattern])
            )
            print(f"\toutput of linear neuron (y): {y_:.2f}")
            print(f"\tsquare error (s.e.): {loss_:.2f}")
            print(
                f"\tweight (w): {np.around(model.w.numpy(), 2)},\tbias (b): {model.b.numpy():.2f}"
            )
            print()

    err.append(np.mean(err_))
    if epoch % 20 == 0:
        print(f"epoch: {epoch}")
        print("\titer: %3d, mse: %1.4f" % (epoch, err[epoch]))

    print(
        f"\tweight (w): {np.around(model.w.numpy(), 2)},\tbias (b): {model.b.numpy():.2f}"
    )

# print learning curve
plt.figure(1)
plt.plot(range(no_epochs), err)
plt.xlabel("epochs")
plt.ylabel("mse")
plt.show()

pred = []
for p in np.arange(len(X)):
    pred.append(model(X[p]).numpy())

print(f"Predicted y: \t{pred}")
print(f"Target y: \t{Y}")
