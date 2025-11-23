import numpy as np

def greedy_score(A, B, window=3):
    # A, B are arrays of shape (m, d) and (n, d)
    m, n = A.shape[0], B.shape[0]
    i = j = 0
    scores = []
    while i < m and j < n:
        best = None
        best_pair = (i,j)
        for di in range(window):
            for dj in range(window):
                ii = min(m-1, i+di)
                jj = min(n-1, j+dj)
                dist = np.linalg.norm(A[ii]-B[jj])
                if best is None or dist < best:
                    best = dist
                    best_pair = (ii, jj)
        scores.append(best)
        i = best_pair[0] + 1
        j = best_pair[1] + 1
    if len(scores)==0:
        return np.inf
    return float(np.mean(scores))
