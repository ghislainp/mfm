
cimport numpy as np
import numpy as np

cdef void solve(Py_ssize_t n, double[:] lower, double[:] diag, double[:] upper,
                double[:] rhs):
    cdef:
        double m
        Py_ssize_t i, im1, nm1 = n - 1

    for i in range(1, n):
        im1 = i - 1
        m = lower[im1] / diag[im1]
        diag[i] -= m * upper[i]
        rhs[i] -= m * rhs[im1]

    diag[nm1] = rhs[nm1] / diag[nm1]
    # diag is x

    for i in range(n - 2, -1, -1):
        # diag is x
        # x[i] = (rhs[i] - upper[i + 1] * x[i + 1]) / diag[i]
        diag[i] = (rhs[i] - upper[i + 1] * diag[i + 1]) / diag[i]


cpdef double[:] tdma(double[:] a, double[:] b, double[:] c,
                     double[:] d):
    cdef:
        Py_ssize_t n = b.shape[0]
        #double[:] x = np.empty(n, dtype=np.float64)
    solve(n, a, b, c, d)
    return b
