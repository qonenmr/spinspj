import numpy as np
import scipy
from scipy.sparse import coo_matrix,csc_matrix,csr_matrix
from scipy.sparse.linalg import splu
import scipy.sparse as sparse
import matplotlib.pyplot as plt

class arpls():
    def __init__(self, data,lamda,ratio):
        self.data = data
        self.lamda = lamda
        self.ratio = ratio
    def speyediff(self, format='csc'):
        N=self.data.shape[0]
        shape = (N - 2, N)
        diagonals = np.zeros(5)
        diagonals[2] = 1.
        for i in range(2):
            diff = diagonals[:-1] - diagonals[1:]
            diagonals = diff
        offsets = np.arange(3)
        spmat = sparse.diags(diagonals, offsets, shape, format=format)
        return spmat
    def fit(self):
        N = self.data.shape[0]
        D = self.speyediff()
        H = D.T.dot(D) * self.lamda
        w = np.ones(N, dtype=float)
        while True:
            col = np.arange(N)
            row = np.arange(N)
            W = csc_matrix((w, (row, col)), shape=(N, N))
            sol = W + H
            z = splu(sol).solve(w * self.data)
            d = self.data - z
            dn = d[d < 0]
            m = np.mean(dn)
            s = np.std(dn)
            wt = 1 / (1 + np.exp((2 * (d - (2 * s - m)) / s)))
            if np.linalg.norm(w - wt) / np.linalg.norm(w) < self.ratio: break
            w = wt
        return z

data=np.loadtxt('system/data/pyexample/Data.txt')
lamda=pow(15,10)
ratio=0.01
z=arpls(data,lamda,ratio).fit()

plt.close('all');
print ("\n"*80)

x = np.linspace(10.7,-1,32768)
plt.gca().invert_xaxis() 
plt.plot(x,data,color='b')
plt.plot(x,z,color='r')
plt.xlabel("ppm")
plt.show()
