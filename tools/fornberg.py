import numpy as np
# ==============================================================================
def fornberg_weights(z, x, m):
    """
    Generates interpolation and finite-differences differentiation weights c^(m)_j
    at point z, provided the functional values have been tabulated at y_j=y(x_j), j=0,n.
    The m-th derivative (m=0 corresponds to interpolation) at z is expressed as
    y^(m)(z) = sum(c^(m)_j * y_j).
    """
    nd = len(x) - 1
    c = np.zeros((nd + 1, m + 1), dtype=float)
    c1 = 1.0
    c4 = x[0] - z
    c[0, 0] = 1.0

    for i in range(1, nd + 1):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - z
        
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        
        c1 = c2

    return c
# ==============================================================================