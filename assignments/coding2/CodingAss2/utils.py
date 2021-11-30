import matplotlib.pyplot as plt
import numpy as np


def plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None):
    X_pos = X[t == 1]
    X_neg = X[t == 0]
    plt.figure()
    plt.scatter(X_pos[:, 0], X_pos[:, 1], color="blue", alpha=0.2)
    plt.scatter(X_neg[:, 0], X_neg[:, 1], color="red", alpha=0.2)

    x_axis = np.linspace(-2, 5, 10)
    y_axis = np.linspace(-2, 5, 10)

    plt.plot(x_axis, [0] * len(x_axis), 'g--')
    plt.plot([0] * len(y_axis), y_axis, 'g--')

    if w is not None:
        x_1 = np.linspace(-2, 5, 10)
        if is_logistic:
            line_fn = lambda x: (- w[0] * x - bias) / w[1]
        else:
            line_fn = lambda x: (- w[0] * x - bias + 0.5) / w[1]
        x_2 = [line_fn(x) for x in x_1]
        plt.plot(x_1, x_2)

    plt.xlabel('x1')
    plt.ylabel('x2')
    ax = plt.gca()

    ax.set_aspect('equal', adjustable='box')

    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    plt.tight_layout()

    if figure_name is None:
        plt.show()
    else:
        plt.savefig(figure_name)


def generate_data(dataset='A'):
    """
    This function generates dataset A or B.
    :param dataset: "A" or "B", case sensitive.
    :return: X, t
    """
    if dataset == 'A':
        shift = False
    elif dataset == 'B':
        shift = True
    else:
        raise ValueError('You should only specify "A" or "B" as the name of the dataset.')

    N = 200
    var = 0.2
    mean = 0.2
    shift_dist = 3
    np.random.seed(1)
    neg_x1 = np.random.normal(-mean, var, N)
    neg_x2 = np.random.normal(-mean, var, N)
    neg_X = np.stack((neg_x1, neg_x2), axis=1)

    N *= 1
    if shift is False:
        pos_x1 = np.random.normal(mean, var, N)
        pos_x2 = np.random.normal(mean, var, int(N / 1))
    else:
        pos_x1_1 = np.random.normal(mean, var, int(N / 2))
        pos_x1_2 = np.random.normal(mean + shift_dist, var, int(N / 2))
        pos_x1 = np.concatenate((pos_x1_1, pos_x1_2))

        pos_x2_1 = np.random.normal(mean, var, int(N / 2))
        pos_x2_2 = np.random.normal(mean + shift_dist, var, int(N / 2))
        pos_x2 = np.concatenate((pos_x2_1, pos_x2_2))

    pos_X = np.stack((pos_x1, pos_x2), axis=1)
    pos_Y = [1] * len(pos_X)
    neg_Y = [0] * len(neg_X)

    all_X = np.concatenate((pos_X, neg_X), axis=0)
    all_t = np.concatenate((pos_Y, neg_Y), axis=0)
    permutation = np.random.permutation(len(all_t))
    all_X = all_X[permutation]
    all_t = all_t[permutation]
    return all_X, all_t

