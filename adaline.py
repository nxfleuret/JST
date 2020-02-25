def bipstep(y, th=0):
    return 1 if y >= th else -1

import numpy as np
import matplotlib.pyplot as plt


def line(w, th=0):
    w2 = w[2] + .001 if w[2] == 0 else w[2]

    return lambda x: (th - (w[1] * x) - w[0]) / w2


def plot(f, s, t):
    x = np.arange(-2, 3)
    col = 'ro', 'bo'

    for c, v in enumerate(np.unique(t)):
        p = s[np.where(t == v)]

        plt.plot(p[:,1], p[:,2], col[c])

    plt.axis([-2, 2, -2, 2])
    plt.plot(x, f(x))
    plt.show()

import sys


def ada_fit(x, t, alpha=.1, max_err=.1, verbose=False, draw=False):
    w = np.random.uniform(0, 1, len(x[0]) + 1)
    b = np.ones((len(x), 1))
    x = np.hstack((b, x))
    stop = False
    epoch = 0

    while not stop:
        epoch += 1
        max_ch = -sys.maxsize

        if verbose:
            print(f'\nEpoch #{epoch}')

        for r, row in enumerate(x):
            y = np.dot(row, w)

            for i in range(len(row)):
                w_new = w[i] + alpha * (t[r] - y) * row[i]
                max_ch = max(abs(w[i] - w_new), max_ch)
                w[i] = w_new

            if verbose:
                print(f'Bobot: {w}')

            if draw:
                plot(line(w), x, t)

        stop = max_ch < max_err

    return w

def ada_predict(X, w):
    y_all = []

    for x in X:
        y_in = w[0] + np.dot(x, w[1:])
        y = bipstep(y_in)

        y_all.append(y)

    return y_all

from sklearn.metrics import accuracy_score

train = (1, 1), (1, -1), (-1, 1), (-1, -1)
target = 1, -1, -1, -1
w = ada_fit(train, target, verbose=True, draw=True)
out = ada_predict(train, w)
accuracy = accuracy_score(out, target)

print(f'Output: {out}')
print(f'Accuracy: {accuracy}')

from sklearn.metrics import accuracy_score

train = (1, 1), (1, -1), (-1, 1), (-1, -1)
target = 1, 1, 1, -1
w = ada_fit(train, target, verbose=True, draw=True)
out = ada_predict(train, w)
accuracy = accuracy_score(out, target)

print(f'Output: {out}')
print(f'Accuracy: {accuracy}')

from sklearn.metrics import accuracy_score

train = (1, 1), (1, -1), (-1, 1), (-1, -1)
target = -1, 1, -1, -1
w = ada_fit(train, target, verbose=True, draw=True)
out = ada_predict(train, w)
accuracy = accuracy_score(out, target)

print(f'Output: {out}')
print(f'Accuracy: {accuracy}')

from sklearn.metrics import accuracy_score

train = (1, 1), (1, -1), (-1, 1), (-1, -1)
target = -1, 1, 1, -1
w = ada_fit(train, target, verbose=True, draw=False)
out = ada_predict(train, w)
accuracy = accuracy_score(out, target)

print(f'Output: {out}')
print(f'Accuracy: {accuracy}')

def mri_fit(X, t, max_err=.1, max_ep=10, draw=False):