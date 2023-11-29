import torch
import numpy as np
import matplotlib.pylab as plt


def plot(X, Y):
    # plot training data
    plt.figure(1)
    plt.plot(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], "bx", label="class A")
    plt.plot(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], "ro", label="class B")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("training data")
    # plt.legend()
    # plt.show()
    return plt


class Logistic:
    def __init__(self) -> None:
        self.w = torch.tensor(np.random.rand(2, 1), dtype=torch.double)
        # self.w = torch.tensor(np.array([[40.5], [2.5]]), dtype=torch.double)
        self.b = torch.tensor(0.0, dtype=torch.double)

    def __call__(self, x):
        u = torch.matmul(torch.tensor(x), self.w) + self.b
        logits = torch.sigmoid(u)  # f(u) = logits
        return u, logits


def loss(targets, logits):
    entropy = -torch.sum(
        torch.tensor(targets) * torch.log(logits)
        + (1 - torch.tensor(targets)) * torch.log(1 - logits)
    )
    classification_error = torch.sum(
        torch.not_equal(logits > 0.5, torch.tensor(targets))
    )
    return entropy, classification_error


def train(model, inputs, targets, learning_rate):
    _, f_u = model(inputs)
    grad_u = -(torch.tensor(targets) - f_u)  # (d - f(u))
    grad_w = torch.matmul(torch.transpose(torch.tensor(inputs), 0, 1), grad_u)
    grad_b = torch.sum(grad_u)

    model.w -= learning_rate * grad_w
    model.b -= learning_rate * grad_b


# training data
# set learning parameters
no_epochs = 3000
lr = 0.01

SEED = 10
np.random.seed(SEED)

X = np.array(
    [
        [5.0, 1.0],
        [7.0, 3.0],
        [3.0, 2.0],
        [5.0, 4.0],
        [0.0, 0.0],
        [-1.0, -3.0],
        [-2.0, 3.0],
        [-3.0, 0.0],
    ]
)
Y = np.array([1, 1, 1, 1, 0, 0, 0, 0]).reshape(8, 1)
# plot(X, Y)

test_cases = np.array(
    [
        [4.0, 2.0],
        [0.0, 5.0],
        [36/13, 0.0]
    ]
)

model = Logistic()

print(f"w: {model.w.numpy()}\nb: {model.b.numpy()}")

entropy, error = [], []
for epoch in range(no_epochs):
    u_, f_u_ = model(X)
    entropy_, err_ = loss(Y, f_u_)

    train(model, X, Y, lr)

    # for first epoch, print valueFs
    # if epoch == 0:
    #     print("u:{}".format(u_))
    #     print("f_u:{}".format(f_u_))
    #     print("y: {}".format((f_u_ > 0.5).numpy().astype(int)))
    #     print("entropy:{}".format(entropy_))
    #     print("error:{}".format(err_))
    #     print("w: {}, b: {}".format(model.w.numpy(), model.b.numpy()))

    if epoch % 100 == 0:
        print("Epoch %2d:  entropy: %2.5f, error: %d" % (epoch, entropy_, err_))
        # print(f"Model w: \n{model.w.numpy()}\nModel b: \n\t{model.b.numpy()}")

    entropy.append(entropy_), error.append(err_)


print(f"Model w: \n{model.w.numpy()}\nModel b: \n\t{model.b.numpy()}")


def formula(x):
    return -(6.5 * x - 14.5) / 2.5


def formula2(x, w, b):
    print(
        f"graph: Model w[0]: \n{w.numpy()[0][0]}, {w.numpy()[1][0]}\nModel b: \n\t{b}"
    )
    return (-b - w.numpy()[0][0] * x) / w.numpy()[1][0]


def graph(x_range, X, Y, x_val, y_val):
    plt = plot(X, Y)
    x1 = np.array(x_range)
    y1 = formula(x1)
    plt.plot(x1, y1, "-")

    y2 = formula2(x1, model.w, model.b)
    plt.plot(x1, y2, "--")

    plt.plot(x_val, y_val, 'bo', linestyle=':')
    plt.ylim(-7, 7)
    plt.show()

class1 = [5.0, 2.5]
class2 = [-1.5, 0.0]
x_vals = [class1[0], class2[0]]
y_vals = [class1[1], class2[1]]

# graph(np.arange(-3, 7.1, 0.1), X, Y, x_vals, y_vals)

def test(x, model):
    # u = (torch.matmul(torch.tensor(x), model.w) + model.b).numpy()
    u = (torch.matmul(torch.tensor(x), torch.tensor(np.array([[6.5], [2.5]]))) -14.5).numpy()
    if u > 0:
        y = 1
    if u <= 0:
        y = 0
    return u[0], y

for p in range(len(X)):
    out, label = test(X[p], model)
    print(f"u: {out:6.2f}\tClass: {label}")
print()
for p in range(len(test_cases)):
    out, label = test(test_cases[p], model)
    print(f"u: {out:6.2f}\tClass: {label}")