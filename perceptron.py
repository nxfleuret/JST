def percep_step(input, th=0):
    return 1 if input > th else 0 if -th <= input <= th else -1

import matplotlib.pyplot as plt

def line(w, th=0):
    w2 = w[2] + .001 if w[2] == 0 else w[2]

    return lambda x: (th - w[1] * x - w[0]) / w2

def plot(f1, f2, s, t):
    x = np.arange(-2, 3)
    col = 'ro', 'bo'

    for c, v in enumerate(np.unique(t)):
        p = s[np.where(t == v)]

        plt.plot(p[:,1], p[:,2], col[c])

    plt.axis([-2, 2, -2, 2])
    plt.plot(x, f1(x))
    plt.plot(x, f2(x))
    plt.show()

import numpy as np

def percep_fit(s, t, th=0, a=1, draw=False):
    # Inisialisasi bobot
    w = np.zeros(len(s[0]) + 1)

    # Inisialisasi bias
    b = np.ones((len(s), 1))

    # Menggabungkan bias dan data latih menjadi satu layer
    s = np.hstack((b, s))

    # Variabel kondisi berhenti
    stop = False
    epoch = 0

    # Lakukan pelatihan selama kondisi berhenti bernilai False
    while not stop:
        stop = True
        epoch += 1

        print(f'\nEpoch #{epoch}')

        for r, row in enumerate(s):
            # Hitung y_in menggunakan dot product
            y_in = np.dot(row, w)

            # Hitung y_out menggunakan fungsi step
            y = percep_step(y_in, th)

            # Jika output tidak sama dengan target
            if y != t[r]:
                stop = False

                # Ubah nilai bobot
                w = [w[i] + a * t[r] * row[i] for i in range(len(row))]

            print(f'Bobot: {w}')

            # Buat grafik jika parameter draw bernilai True
            if draw:
                plot(line(w, th), line(w, -th), s, t)

    return w

def percep_predict(x, w, th=0):
    y_in = w[0] + np.dot(x, w[1:])

    return percep_step(y_in, th)

#Logika AND
train = [1, 1], [1, -1], [-1, 1], [-1, -1]
target = 1, -1, -1, -1
th = .2
w = percep_fit(train, target, th, draw=True)

print(percep_predict([1, 1], w, th))

#Logika OR
train = [1, 1], [1, -1], [-1, 1], [-1, -1]
target = 1, 1, 1, -1
w = percep_fit(train, target, draw=True)

print(percep_predict([1, 1], w))