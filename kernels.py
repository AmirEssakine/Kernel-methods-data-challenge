import numpy as np
from collections import Counter
from numba import njit, prange
from tqdm import tqdm

class SpectrumKernel:
    def __init__(self, k=3):
        self.k = k
        
        
    def make_phi(self, seq, k=None):
        assert (len(seq) >= k)
        dict_ = {}
        for i in range(len(seq) - k + 1):
            target = tuple(seq[i : i + k])
            if target not in dict_:
                dict_[target] = 1
            else:
                dict_[target] += 1
        return dict_


    def kernel(self, seq1, seq2):
        dict1 = self.make_phi(seq1, self.k)
        dict2 = self.make_phi(seq2, self.k)
    
        keys = list(set(dict1.keys()) & set(dict2.keys()))
        output = 0
        for key in keys:
            output += dict1[key] * dict2[key]
        return output


class MismatchKernel:
    def __init__(self, k, m):
        self.k = k
        self.m = m

    def hamming(self, s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def kernel(self, seq1, seq2):
        combinations = [(i, d) for i in range(len(seq1)-self.k+1) for d in range(len(seq2)-self.k+1)]
        return sum(self.hamming(seq1[i:i+self.k], seq2[d:d+self.k]) <= self.m
                   for i, d in combinations)
        
class ConvNgramKernel:
    def __init__(self, gamma=10.0, n=8, k=4, M=4096):
        self.gamma = gamma
        self.n = n
        self.k = k
        self.M = M

    def one_hot_encode(self, sequences):
        mapping = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0], "G": [0, 0, 1, 0], "C": [0, 0, 0, 1]}
        encoded = np.array([[mapping[char] for char in seq] for seq in sequences], dtype=np.float32)
        return encoded.reshape(len(sequences), -1)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def compute_features(X, ws, bs, start, end, M, n, k):
        batch_size = end - start
        d = X.shape[1] // k
        phi_batch = np.zeros((batch_size, M), dtype=np.float32)

        for i in prange(batch_size):
            seq_idx = start + i
            for j in prange(M):
                z = np.zeros(d - n + 1, dtype=np.float32)

                for t in range(d - n + 1):
                    X_sub = X[seq_idx, t * k : (t + n) * k]
                    z[t] = np.dot(X_sub, ws[j])

                phi_batch[i, j] = np.sqrt(2.0 / M) * np.sum(np.cos(z + bs[j]))

        return phi_batch

    def map(self, sequences):
        X = np.ascontiguousarray(self.one_hot_encode(sequences))
        N = X.shape[0]

        batch_size = 1

        ws = np.random.randn(self.M, self.k * self.n).astype(np.float32) * np.sqrt(self.gamma)
        bs = np.random.uniform(0, 2 * np.pi, self.M).astype(np.float32)

        phi = np.zeros((N, self.M), dtype=np.float32)

        for start in tqdm(range(0, N, batch_size), desc="Computing features", unit="batch"):
            end = min(start + batch_size, N)
            phi[start:end] = self.compute_features(X, ws, bs, start, end, self.M, self.n, self.k)

        return phi

    def kernel(self, sequences1, sequences2):
        phi_X = self.map(sequences1)
        if sequences1 is sequences2:
            phi_Y = phi_X
        else:
            phi_Y = self.map(sequences2)

        N, M = phi_X.shape
        P, _ = phi_Y.shape
        gram_matrix = np.zeros((N, P), dtype=np.float32)

        for i in tqdm(range(N), desc="Computing Gram Matrix", unit="row"):
            for j in range(P):
                gram_matrix[i, j] = np.dot(phi_X[i], phi_Y[j])

        return gram_matrix

class GappedKmerKernel:
    def __init__(self, k, gap):
        self.k = k
        self.gap = gap

    def kernel(self, seq1, seq2):
        subseq_set = set(seq1[i:i+self.k] for i in range(len(seq1) - self.k + 1))
        return sum(seq2[i:i+self.k] in subseq_set for i in range(0, len(seq2) - self.k + 1, self.gap))

class WeightedDegreeKernel:
    def __init__(self, k):
        self.k = k

    def kernel(self, seq1, seq2):
        kernel_value = 0
        for i in range(len(seq1) - self.k + 1):
            kernel_value += sum(seq1[i:i+self.k] == seq2[j:j+self.k] for j in range(len(seq2)-self.k+1))
        return kernel_value

class PolynomialKernel:
    def __init__(self, degree=3):
        self.degree = degree

    def kernel(self, x, y):
        return (np.dot(x, y) + 1) ** self.degree