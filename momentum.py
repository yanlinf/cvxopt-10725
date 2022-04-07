import argparse
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

COLORS = ['#5b9bd5', '#dba58c', '#99cccc', '#ff8b8b', '#b98fc4']

TABLE_LATEX = '''
\\begin{{center}}
\\begin{{tabular}}{{|c|c|c|c|c|c|}}
    \hline
    \(\gamma \downarrow / \eta \\rightarrow\) & 0.01 & 0.1 & 1 & 10 & 100 \\\\
    \hline
    0.01 & {:.6f} & {:.6f} & {:.6f} & {:.6f} & {:.6f} \\\\
    0.1 & {:.6f} & {:.6f} & {:.6f} & {:.6f} & \(\\times\) \\\\
    1 & {:.6f} & {:.6f} & {:.6f} & \(\\times\) & \(\\times\) \\\\
    \hline
\end{{tabular}}
\end{{center}}
'''

TABLE2 = '''
\\begin{{center}}
\\begin{{tabular}}{{|c|c|c|c|c|}}
    \hline
    \(b \downarrow / \eta \\rightarrow\) & 1 & 0.3 & 0.1 & 0.01 \\\\
    \hline
    1 & {:.6f} & {:.6f} & {:.6f} & {:.6f} \\\\
    10 & {:.6f} & {:.6f} & {:.6f} & {:.6f} \\\\
    100 & {:.6f} & {:.6f} & {:.6f} & {:.6f} \\\\
    1000 & {:.6f} & {:.6f} & {:.6f} & {:.6f} \\\\
    \hline
\end{{tabular}}
\end{{center}}
'''


def func(X, y, w):
    return np.log(np.exp((-y) * X.dot(w)) + 1).mean()


def grad(X, y, w):
    return X.T.dot(1 / (1 + np.exp(y * X.dot(w))) * (-y)) / X.shape[0]


def sgd(X, y, w0, lr, bs, n_iter, seed=0):
    np.random.seed(seed)
    n, d = X.shape
    w = w0
    his_f = []
    his_w = []
    for i in range(n_iter):
        idx = np.random.choice(n, size=bs, replace=True)
        w = w - lr * grad(X[idx], y[idx], w)
        his_f.append(func(X, y, w))
        his_w.append(w)
    return w, his_f, his_w


def momentum(X, y, w0, lr, gamma, n_iter):
    n, d = X.shape
    w = w0
    g = np.zeros(d)
    his_f = []
    his_w = []
    for i in range(n_iter):
        g = (1 - gamma) * g + gamma * grad(X, y, w)
        w = w - lr * g  # w_{i+1}
        his_f.append(func(X, y, w))
        his_w.append(w)
    return w, his_f, his_w


def n_sgd(X, y, w0, lr, bs, n_iter, n_trial=25):
    res = []
    his_f_all = []
    his_w_all = []
    for seed in range(n_trial):
        w, his_f, his_w = sgd(X, y, w0, lr, bs, n_iter, seed=seed)
        res.append(func(X, y, w))
        his_f_all.append(his_f)
        his_w_all.append(his_w)
    return sum(res) / n_trial, np.array(his_f_all).mean(0), np.array(his_w_all).mean(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='data/samples.csv')
    args = parser.parse_args()

    with open(args.input, 'r') as fin:
        data = np.array([[float(x) for x in line.rstrip().split(',')] for line in fin])

    X = data[:, :-1]
    y = data[:, -1]  # dtype=float

    n, d = X.shape
    print(f'n={n}  d={d}')
    w = cp.Variable(d)
    loss = cp.sum(cp.logistic(cp.multiply(-y, (X @ w)))) / n

    # Q2.1
    prob = cp.Problem(cp.Minimize(loss))
    f_opt = prob.solve()
    w_opt = w.value
    print(f'optimal_np: {func(X, y, w.value):.12f}')
    print(f'optimal_cvxpy: {f_opt:.12f}')
    print()

    # Q2.2
    params = {
        0.01: [0.01, 0.1, 1, 10, 100],
        0.1: [0.01, 0.1, 1, 10],
        1: [0.01, 0.1, 1],
    }
    w0 = np.zeros(d)
    n_iter = 1000

    fig = plt.figure()
    results = []
    for i, gamma in enumerate([0.01, 0.1, 1]):
        ax = fig.add_subplot(1, 3, i + 1)
        for j, lr in enumerate(params[gamma]):
            w, his_f, his_w = momentum(X, y, w0, lr, gamma, n_iter=n_iter)
            his_f = np.array(his_f)
            results.append(func(X, y, w))
            ax.plot(np.arange(n_iter), his_f - f_opt, label=f'lr={lr}', color=COLORS[j])
            f = func(X, y, w)
            print(f'gamma={gamma}  lr={lr} f*={f}')
        ax.set_yscale('log')
        ax.set_xlabel('n_iter')
        ax.set_title(f'gamma={gamma}')
        if i == 0:
            ax.set_ylabel('log(f-f*)')
        ax.legend()
        ax.grid()
        ax.spines['bottom'].set_color('lightgray')
        ax.spines['top'].set_color('lightgray')
        ax.spines['right'].set_color('lightgray')
        ax.spines['left'].set_color('lightgray')
        ax.tick_params(length=0)
    fig.set_size_inches(12, 4)
    fig.savefig('momentum.pdf', format='pdf', bbox_inches='tight')
    print()

    # Q2.3
    print(TABLE_LATEX.format(*results))

    # Q2.5
    fig = plt.figure()
    results = []
    for i, gamma in enumerate([0.01, 0.1, 1]):
        ax = fig.add_subplot(1, 3, i + 1)
        for j, lr in enumerate(params[gamma]):
            w, his_f, his_w = momentum(X, y, w0, lr, gamma, n_iter=n_iter)
            his_w = np.array(his_w)
            results.append(func(X, y, w))
            ax.plot(np.arange(n_iter), np.linalg.norm(his_w - w_opt, axis=1), label=f'lr={lr}', color=COLORS[j])
            f = func(X, y, w)
            print(f'gamma={gamma}  lr={lr} f*={f}')
        # ax.set_yscale('log')
        ax.set_xlabel('n_iter')
        ax.set_title(f'gamma={gamma}')
        if i == 0:
            ax.set_ylabel('|wt-w*|')
        ax.legend()
        ax.grid()
        ax.spines['bottom'].set_color('lightgray')
        ax.spines['top'].set_color('lightgray')
        ax.spines['right'].set_color('lightgray')
        ax.spines['left'].set_color('lightgray')
        ax.tick_params(length=0)
    fig.set_size_inches(12, 4)
    fig.savefig('Q2-5.pdf', format='pdf', bbox_inches='tight')
    print()

    # Q3.1
    w = cp.Variable(d)
    loss = cp.sum(cp.logistic(cp.multiply(-y, (X @ w)))) / n
    prob = cp.Problem(cp.Minimize(loss))
    f_opt = prob.solve()
    w_opt = w.value
    print(f'optimal_np: {func(X, y, w.value):.12f}')
    print(f'optimal_cvxpy: {f_opt:.12f}')
    print()

    w0 = np.zeros(d)
    n_iter = 500
    f_all = {}
    his_f_all = {}
    his_w_all = {}
    for bs in [1, 10, 100, 1000]:
        for lr in [1, 0.3, 0.1, 0.03]:
            f, his_f, his_w = n_sgd(X, y, w0, lr, bs, n_iter=n_iter, n_trial=25)
            print(f'lr={lr} bs={bs} f*={f}')
            f_all[(bs, lr)] = f
            his_f_all[(bs, lr)] = his_f
            his_w_all[(bs, lr)] = his_w

    # Q3.2
    results = []
    for bs in [1, 10, 100, 1000]:
        for lr in [1, 0.3, 0.1, 0.03]:
            results.append(f_all[(bs, lr)])
    print(TABLE2.format(*results))
    print()

    # Q3.3
    fig = plt.figure()
    for i, lr in enumerate([1, 0.3, 0.1, 0.03]):
        ax = fig.add_subplot(1, 4, i + 1)
        for j, bs in enumerate([1, 10, 100, 1000]):
            his_w = np.array(his_w)
            ax.plot(np.arange(n_iter), his_f_all[(bs, lr)] - f_opt, label=f'b={bs}', color=COLORS[j])
        ax.set_yscale('log')
        ax.set_xlabel('n_iter')
        ax.set_title(f'lr={lr}')
        if i == 0:
            ax.set_ylabel('log(ft-f*)')
        ax.legend()
        ax.grid()
        ax.spines['bottom'].set_color('lightgray')
        ax.spines['top'].set_color('lightgray')
        ax.spines['right'].set_color('lightgray')
        ax.spines['left'].set_color('lightgray')
        ax.tick_params(length=0)
    fig.set_size_inches(16, 4)
    fig.savefig('Q3-3.pdf', format='pdf', bbox_inches='tight')
    print()

    # Q3.4
    fig = plt.figure()
    for i, bs in enumerate([1, 10, 100, 1000]):
        ax = fig.add_subplot(1, 4, i + 1)
        for j, lr in enumerate([1, 0.3, 0.1, 0.03]):
            his_w = np.array(his_w)
            ax.plot(np.arange(n_iter), his_f_all[(bs, lr)] - f_opt, label=f'lr={lr}', color=COLORS[j])
        ax.set_yscale('log')
        ax.set_xlabel('n_iter')
        ax.set_title(f'b={bs}')
        if i == 0:
            ax.set_ylabel('log(ft-f*)')
        ax.legend()
        ax.grid()
        ax.spines['bottom'].set_color('lightgray')
        ax.spines['top'].set_color('lightgray')
        ax.spines['right'].set_color('lightgray')
        ax.spines['left'].set_color('lightgray')
        ax.tick_params(length=0)
    fig.set_size_inches(16, 4)
    fig.savefig('Q3-4.pdf', format='pdf', bbox_inches='tight')
    print()


if __name__ == '__main__':
    main()
