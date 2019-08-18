import numpy as np
import math


def entropy(y):
    if len(y) == 0:
        return 0
    p1 = np.count_nonzero(y) / float(len(y))
    p0 = 1-p1

    return -p0*np.log2(max(1e-10, 1-p1)) -p1*np.log2(max(1e-10, np.count_nonzero(y) / float(len(y))))


def jointEntropy(x1, x2, y):
    tot_entropy = 0
    for i in [0,1]:
        for j in [0,1]:
            (x1_idx,) = np.where(x1 == i)
            (x2_idx,) = np.where(x2 == j)

            idx = np.intersect1d(x1_idx, x2_idx)

            sub_set_y = y[idx]
            w = (len(idx)/len(y))
            e = entropy(sub_set_y)

            tot_entropy += e*w

    # print('Total entropy: ', tot_entropy)
    return tot_entropy


y = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])

print('check')
# case A
x1 = np.zeros(16)
x2 = np.zeros(16)
entropy1 = jointEntropy(x1,x2,y)
#assert(entropy1==1, 'wrong value for case A')
print(entropy1)
# case B
x1 = np.zeros(16)
x2 = y.copy()
entropy2 = jointEntropy(x1,x2,y)
#assert(entropy2==0, 'wrong value for case B')
print(entropy2)

# case C
x1 = np.ones(16)
x2 = np.array([0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0])
entropy3 = jointEntropy(x1,x2,y)
#assert(entropy3==1, 'wrong value for case C')
print(entropy3)
# case D
x1 = np.zeros(16)
x2 = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
entropy4 = jointEntropy(x1,x2,y)
#assert(entropy4==1, 'wrong value for case D')
print(entropy4)
# case E
x1 = np.array([0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0])
x2 = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
entropy5 = jointEntropy(x1,x2,y)
#assert(entropy5==0, 'wrong value for case E')
print(entropy5)
# case F
x1 = np.array([1,1,1,1,0,1,0,1,1,0,1,0,1,0,1,0])
x2 = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
entropy6 = jointEntropy(x1,x2,y)
difference = entropy6-0.344
assert(np.abs(difference) < 0.01, 'wrong value for case F')
#print(entropy6)

