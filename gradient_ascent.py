import argparse
import numpy as np
import matplotlib.pyplot as plt

COLORS = ['#5b9bd5', '#dba58c', '#99cccc', '#ff8b8b', '#b98fc4']


def gradient_ascent(func, grad, x0, lr, n_iter):
    his_f, his_x = [func(x0)], [x0]
    x = x0
    for i in range(n_iter):
        x += lr * grad(x)
        his_f.append(func(x))
        his_x.append(x)
    return x, his_f, his_x


def set_axis_style(ax):
    ax.grid()
    ax.spines['bottom'].set_color('lightgray')
    ax.spines['top'].set_color('lightgray')
    ax.spines['right'].set_color('lightgray')
    ax.spines['left'].set_color('lightgray')
    ax.tick_params(length=0)


def main():
    func = lambda x: x + np.sin(6 * x)
    grad = lambda x: 1 + 6 * np.cos(6 * x)
    results = {}
    step_sizes = [0.01, 0.02, 0.05]
    n_iter = 100
    x0 = 0
    for lr in step_sizes:
        x, his_f, his_x = gradient_ascent(func, grad, 0, lr, n_iter)
        results[lr] = (x, his_f, his_x)

    x_opt = (np.arccos(-1/6) + np.pi * 8) / 6
    f_opt = func(x_opt)
    print(f'x* = {x_opt}')
    print(f'f* = {f_opt}')

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    for i, lr in enumerate(step_sizes):
        his_f = np.array(results[lr][1])
        ax.plot(np.arange(n_iter + 1), f_opt - his_f, label=f'lr={lr}', color=COLORS[i])
    ax.set_xlabel('n_iter')
    ax.set_ylabel('f* - f(xt)')
    ax.legend()
    set_axis_style(ax)

    ax = fig.add_subplot(1, 2, 2)
    for i, lr in enumerate(step_sizes):
        his_x = np.array(results[lr][2])
        ax.plot(np.arange(n_iter + 1), np.abs(x_opt - his_x), label=f'lr={lr}', color=COLORS[i])
    ax.set_yscale('log')
    ax.set_xlabel('n_iter')
    ax.set_ylabel('log(|xt - x*|)')
    ax.legend()
    set_axis_style(ax)

    fig.set_size_inches(10, 4)
    fig.savefig('Q5-1.pdf', format='pdf', bbox_inches='tight')
    print('Q5-1.pdf saved')


if __name__ == '__main__':
    main()
