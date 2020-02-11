import numpy as np

#fungsi step biner
def binstep(x, th):
    return 1 if x >= th else 0

# n = [1,2]
# w = [3,4]
# y = np.dot(n,w)

# print(y)
# print(binstep(y))

x = [0, 0]

#step biner logika AND
def AND(x):
    w = [1,1]
    y_in = np.dot(x, w)

    return binstep(y_in, 2)

# print(AND(x))

# step biner logika OR
def OR(x):
    w = [2,2]
    y_in = np.dot(x, w)

    return binstep(y_in, 1)

# print(OR(x))

def ANDNOT1(x):
    w = [2,-1]
    y_in = np.dot(x, w)

    return binstep(y_in, 2)

# print(ANDNOT(x))

def ANDNOT2(x):
    w = [-1,2]
    y_in = np.dot(x, w)

    return binstep(y_in, 2)

def XOR(x):
    wOR = [2,2]
    z = [ANDNOT1(x), ANDNOT2(x)]
    y_in = np.dot(z, wOR)

    return binstep(y_in, 2)

print(XOR(x))