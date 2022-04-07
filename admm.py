import numpy as np
import matplotlib.pyplot as plt

COLORS = ['#5b9bd5', '#dba58c', '#99cccc', '#ff8b8b', '#b98fc4']


def admm(w, a1, a2, lr):
    w1_his, w2_his, a1_his, a2_his = [], [], [], []
    step = 0
    while True:
        a1_his.append(a1)
        a2_his.append(a2)
        w_prev = w
        w1 = 0.5 * w - 0.25 * a1 + 1
        w2 = 0.25 * w - 0.125 * a2 - 1
        w1_his.append(w1)
        w2_his.append(w2)
        w = (a1 + a2) / 4 + (w1 + w2) / 2
        a1 = a1 - lr * (w - w1)
        a2 = a2 - lr * (w - w2)
        print(f'step: {step}  w: {w}  a1: {a1}  a2: {a2}')
        if step > 10 and abs(w - w_prev) < 1e-4:
            break
        step += 1
    return w, a1, a2, w1_his, w2_his, a1_his, a2_his


def set_axis_style(ax):
    ax.grid()
    ax.spines['bottom'].set_color('lightgray')
    ax.spines['top'].set_color('lightgray')
    ax.spines['right'].set_color('lightgray')
    ax.spines['left'].set_color('lightgray')
    ax.tick_params(length=0)


def main():
    w, a1, a2, w1_his, w2_his, a1_his, a2_his = admm(0, 0, 0, 2)

    xs = np.arange(len(a1_his))
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(xs, a1_his, color=COLORS[0], label='a1')
    ax.plot(xs, a2_his, color=COLORS[1], label='a2')
    ax.legend()
    set_axis_style(ax)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(xs, w1_his, color=COLORS[2], label='w1')
    ax.plot(xs, w2_his, color=COLORS[3], label='w2')
    ax.legend()
    set_axis_style(ax)

    fig.set_size_inches(8, 4)
    fig.savefig('admm.pdf', format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
