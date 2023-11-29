import torch
import numpy as np
import matplotlib.pylab as plt


class Softmax:
    def __init__(self, no_features, no_labels):
        self.w = torch.tensor(
            np.array([[0.88, 0.08, -0.34], [0.68, -0.39, -0.19]]), dtype=torch.double
        )
        self.b = torch.zeros([no_labels], dtype=torch.double)

    def __call__(self, x):
        """
        Args: Input features in matrix X
        Returns U, f(U), Y
        """
        u = torch.matmul(torch.tensor(x), self.w) + self.b
        fu = torch.exp(u) / torch.sum(torch.exp(u), dim=1, keepdims=True)
        y = torch.argmax(fu, dim=1)  # Y here is calculated Y, not target labels d okay
        return u, fu, y


def loss(fu, k, y):
    """
    entropy: f(U) multiplies by one hot vector-ed matrix to make non max f(u) = 0.
    error: check if argmax of each row of matrix K is the same as target label Y.
    """
    entropy = -torch.sum(torch.log(fu) * k)
    error = torch.sum(torch.not_equal(torch.argmax(k, dim=1), y))
    return entropy, error


def train(model, inputs, k, learning_rate):
    _, fu, y = model(inputs)
    grad_u = -(k - fu)
    grad_w = torch.matmul(torch.transpose(torch.tensor(inputs), 0, 1), grad_u)
    grad_b = torch.sum(grad_u, dim=0)

    model.w -= learning_rate * grad_w
    model.b -= learning_rate * grad_b
    return grad_u, grad_w, grad_b


def plot_err_entropy(err, entropy, no_epochs):
    plt.figure(2)
    plt.plot(range(no_epochs), entropy)
    plt.xlabel("Epochs")
    plt.ylabel("Multi-Class Cross Entropy")
    plt.show()

    plt.figure(3)
    plt.plot(range(no_epochs), err)
    plt.xlabel("Epochs")
    plt.ylabel("Classification Errors")
    plt.show()


def plot_data(X, Y):
    plt.figure(1)
    plt.plot(X[Y == 0, 0], X[Y == 0, 1], "b^", label="class A")
    plt.plot(X[Y == 1, 0], X[Y == 1, 1], "ro", label="class B")
    plt.plot(X[Y == 2, 0], X[Y == 2, 1], "gx", label="class C")

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()

def plot_decision_boundaries(X, Y, model):
    w, b = model.w.numpy(), model.b.numpy()
    ww, bb = np.zeros((3, 2)), np.zeros(3)
    for i in range(3):
        ww[i, :] = w[:, i] - w[:, (i+1)%3]
        bb[i] = b[i] - b[(i+1)%3]

    print('ww: {}'.format(ww))
    print('bb: {}'.format(bb))
    
    m = -ww[:,0]/ww[:,1]
    c = -bb/ww[:,1]

    print('m: {}'.format(m))
    print('c: {}'.format(c))

    def compute_line(x):
        y = np.zeros((3, x.shape[0]))
        for i in range(3):
            y[i] = m[i]*x + c[i]
        return y

    xx = np.arange(-4.5, 4.5, 0.01)
    yy = compute_line(xx)
    print(f"xx:\n{xx}")
    print(f"yy:\n{yy}")
    plt.figure(4)
    line = plt.subplot(1, 1, 1)
    line.plot(xx[yy[0] > yy[2]], yy[0, yy[0] > yy[2]], color = 'blue', linestyle = '-')
    line.plot(xx[yy[1] < yy[2]], yy[1, yy[1] < yy[2]], color = 'red', linestyle = '-')
    line.plot(xx[yy[2] > yy[0]], yy[2, yy[2] > yy[0]], color = 'green', linestyle = '-')
    plot_pred = line.plot(X[Y==0,0], X[Y==0,1], 'b^', label='class A')
    plot_original = line.plot(X[Y==1,0], X[Y==1,1], 'ro', label='class B')
    plot_original = line.plot(X[Y==2,0], X[Y==2,1], 'gx', label='class C')
    ax = plt.gca()  # gca stands for 'get current axis'
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    plt.ylim(-4.5, 4.5)
    plt.xlim(xx.min(), xx.max())
    plt.text(4.3, 0.1, r'$x_1$')
    plt.text(0.1, 4.3, r'$x_2$')
    plt.text(-5.25, 2.25, r'$Class A = {:.2f}x_1+{:.2f}x_2-{:.2f}=0$'.format(ww[0][0], ww[0][1], bb[0]))
    plt.text(-5.25, -3.0, r'$Class B = {:.2f}x_1+{:.2f}x_2-{:.2f}=0$'.format(ww[1][0], ww[1][1], bb[1]))
    plt.text(0.5, 3.5, r'$Class C = {:.2f}x_1+{:.2f}x_2-{:.2f}=0$'.format(ww[2][0], ww[2][1], bb[2]))
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": "{:12.8f}".format})

    no_epochs = 3000
    no_inputs = 2
    no_classes = 3
    learning_rate = 0.05

    SEED = 10
    np.random.seed(SEED)

    X = np.array(
        [
            [0.0, 4.0],
            [-1.0, 3.0],
            [2.0, 3.0],
            [-2.0, 2.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [-1.0, 2.0],
            [-3.0, 1.0],
            [-1.0, 1.0],
            [2.0, 1.0],
            [4.0, 1.0],
            [-2.0, 0],
            [1.0, 0.0],
            [3.0, 0.0],
            [-3.0, -1.0],
            [-2.0, -1.0],
            [2.0, -1.0],
            [4.0, -1.0],
        ]
    )

    Y = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2]).astype(int)

    K = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )

    # Plotting data:
    # plot_data(X, Y, decision_boundaries=False, model=None)

    model = Softmax(no_inputs, no_classes)
    print(f"Model W:\n{model.w}")
    print(f"Model b:\n{model.b}")

    loss_ = []
    error_ = []

    for epoch in range(no_epochs):
        u_, fu_, y_ = model(X)
        entropy, err = loss(fu_, torch.tensor(K), y_)
        loss_.append(entropy)
        error_.append(err)
        grad_u, grad_w, grad_b = train(model, X, torch.tensor(K), learning_rate)
        if epoch == 0:
            print(f"Epoch: {epoch + 1}")
            print(f"W:\n{model.w.numpy()}")
            print(f"b:\n{model.b.numpy()}")
            print(f"U:\n{u_.numpy()}")
            print(f"f(U):\n{fu_.numpy()}")
            print(f"\nOriginal Y:\n{Y}")
            print(f"Predicted Y:\n{y_.numpy()}")
            print(f"Entropy:\n{entropy.numpy()}")
            print(f"Error:\n{err.numpy()}")
            print(f"Gradient of J w.r.t. U:\n{grad_u.numpy()}")
            print(f"Gradient of J w.r.t. W:\n{grad_w.numpy()}")
            print(f"Gradient of J w.r.t. b:\n{grad_b.numpy()}")

        if epoch > (no_epochs - 10):
            print(f"\n\nEpoch: {epoch + 1}")
            print(f"Model.W:\n{model.w}")
            print(f"Model.b:\n{model.b}")
            print(f"After activation - f(U):\n{fu_.numpy()}")
            print(f"\nOriginal Y:\n{Y}")
            print(f"Predicted Y:\n{y_.numpy()}")

    # plot_err_entropy(error_, loss_, no_epochs)
    plot_decision_boundaries(X, Y, model)
