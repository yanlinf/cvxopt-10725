import argparse
import numpy as np
import cvxpy as cp

def func(X, y, w):
    return np.log(np.exp((-y) * X.dot(w)) + 1).mean()


def grad(X, y, w):
    return X.T.dot(1 / (1 + np.exp(y * X.dot(w))) * (-y)) / X.shape[0]

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
    print()


if __name__ == '__main__':
    main()
