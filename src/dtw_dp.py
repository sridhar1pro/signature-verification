# src/dtw_dp.py
import numpy as np

def dtw_distance(A, B, band=None, large_cost=1e6):
    m, n = len(A), len(B)
    # guard: if empty
    if m == 0 or n == 0:
        return float('inf')

    D = np.full((m+1, n+1), np.inf, dtype=float)
    D[0,0] = 0.0

    if band is None:
        jmin_func = lambda i: 1
        jmax_func = lambda i: n
    else:
        jmin_func = lambda i: max(1, i - band)
        jmax_func = lambda i: min(n, i + band)

    for i in range(1, m+1):
        jmin = jmin_func(i)
        jmax = jmax_func(i)
        for j in range(jmin, jmax+1):
            cost = np.linalg.norm(A[i-1] - B[j-1])
            if not np.isfinite(cost):
                cost = large_cost
            # accumulate
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])

    dist = D[m,n]
    if not np.isfinite(dist):
        return float('inf')
    # normalize by path length approximation (m+n)
    return float(dist) / (m + n)
