# -*- coding: utf-8 -*-
#
#    MFM
#    Copyright (C) 2013-2018 Ghislain Picard
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import scipy.linalg

# make numba import optional
try:
    from numba import jit
    from numba import float64 as nfloat64

    default_solver = 'directsolver'  # could use fortran solver but it is only marginally faster.

except ImportError:
    # numba is not available
    print("numba is not available. It is recommended to install numba.")

    def jit(*args, **kwargs):
        return lambda f, *k: f
    nfloat64 = []

    default_solver = 'scipy'


def cranknicolson_dirichlet(temp, tsurf, dt, dx, ks, rho, cpice=2092.0, tbase=None):
    ab, b = compiled_cranknicolson_dirichlet(temp, tsurf, dt, dx, ks, rho, cpice=cpice, tbase=tbase)
    return scipy.linalg.solve_banded((1, 1), ab, b)


@jit((nfloat64[:], nfloat64, nfloat64, nfloat64[:], nfloat64[:], nfloat64[:], nfloat64[:], nfloat64), nopython=True, cache=True)
def compiled_cranknicolson_dirichlet(temp, tsurf, dt, dx, ks, rho, cpice=2092.0, tbase=None):

    n = len(temp)
    ab = np.zeros((3, n))
    b = np.zeros(n)

    # a faire une seule fois
    dx_ks = dx / ks
    beta = np.empty(n)
    beta[1:] = 1 / (dx_ks[0:-1] + dx_ks[1:])
    beta[0] = 0

    beta1 = beta[:-1]
    beta2 = beta[1:]

    # intermediate layers

    timeterm = dx * rho * cpice / dt

    # diagonal
    ab[1, 0:-1] = timeterm[0:-1] + beta1 + beta2
    # upperdiag
    ab[0, 1:] = -beta2
    # ab[0,1:]=-beta1

    # lowerdiag
    # in principle it's beta1, but as lowerdiag is shifted by one step, shifted beta1 is beta2
    ab[2, :-1] = -beta2

    # top layer # dirichlet
    ab[1, 0] = 1.0  # diag
    ab[0, 1] = 0.0  # upperdiag

    if tbase is None:
        # bottom layer # flux nul
        ab[1, -1] = timeterm[-1] + beta2[-1]  # diag
        ab[2, -2] = -beta2[-1]  # lowerdiag
    else:
        # bottom layer # Dirichlet en bas:
        ab[1, -1] = 1.0
        ab[2, -2] = 0.0

    # top layer
    b[0] = tsurf

    if tbase is None:
        # bottom layer # flux nul
        b[-1] = (timeterm[-1] - beta2[-1]) * temp[-1] + beta2[-1] * temp[-2]
    else:
        # si Dirichlet en bas:
        b[-1] = tbase

    # intermediate layers
    b[1:-1] = (timeterm[1:-1] - beta1[1:] - beta2[1:]) * \
        temp[1:-1] + beta1[1:] * temp[0:-2] + beta2[1:] * temp[2:]

    return ab, b


def cranknicolson_neuman(temp, flux, dflux, dt, dx, ks, rho, cpice=2092.0, tbase=273, solver=None):

    if solver is None:
        # depends on whether numba is available or not
        solver = default_solver

    if solver == "directsolver":
        return compiled_cranknicolson_neuman_directsolver(temp, flux, dflux, dt, dx, ks, rho, cpice, tbase)

    if solver == "fortran":
        from cranknicholson_f import cranknicholsonneuman
        workspace = np.empty(len(temp) * 5)
        cranknicholsonneuman(temp, flux, dflux, dt, dx, ks, rho, cpice, tbase, True, workspace)
        return temp.copy()

    if solver.startswith("loop_"):
        solver = solver[5:]
        ab, b = compiled_cranknicolson_neuman_loop(temp, flux, dflux, dt, dx, ks, rho, cpice, tbase)
    else:
        ab, b = compiled_cranknicolson_neuman(temp, flux, dflux, dt, dx, ks, rho, cpice, tbase)

    if solver == 'scipy':
        return scipy.linalg.solve_banded((1, 1), ab, b)
    elif solver in 'tdma':
        return TDMAsolver(ab[2], ab[1], ab[0], b)
    elif solver == 'c_tdma':
        from .tdma import tdma   # require cython compilation.
        return np.asarray(tdma(ab[2], ab[1], ab[0], b))
    else:
        raise RuntimeError("unknown solver '%s'" % solver)


@jit((nfloat64[:], nfloat64, nfloat64, nfloat64, nfloat64[:], nfloat64[:], nfloat64[:], nfloat64[:], nfloat64), nopython=True, cache=True)
def compiled_cranknicolson_neuman(temp, flux, dflux, dt, dx, ks, rho, cpice=2092.0, tbase=273):

    n = len(temp)
    ab = np.empty((3, n))
    b = np.empty(n)

    # a faire une seule fois
    dx_ks = dx / ks
    beta = np.empty(n)
    beta[1:] = 1 / (dx_ks[:-1] + dx_ks[1:])
    beta[0] = 0

    beta1 = beta[:-1]
    beta2 = beta[1:]

    # intermediate layers

    timeterm = dx * rho * cpice / dt

    # diagonal
    ab[1, 1:-1] = timeterm[1:-1] + beta1[1:] + beta2[1:]
    # upperdiag
    ab[0, 1:] = -beta2

    # lowerdiag
    # in principle it's beta1, but as lowerdiag is shifted by one step, shifted beta1 is beta2
    ab[2, :-1] = -beta2

    # top layer
    ab[1, 0] = timeterm[0] + beta2[0] - 0.5 * dflux  # diag
    b[0] = (timeterm[0] - beta2[0] - 0.5 * dflux) * temp[0] + beta2[0] * temp[1] + flux

    if tbase is None:
        # bottom layer # flux nul
        ab[1, -1] = timeterm[-1] + beta2[-1]   # diag
        ab[2, -2] = -beta2[-1]  # lowerdiag
        # bottom layer # flux nul
        b[-1] = (timeterm[-1] - beta2[-1]) * temp[-1] + beta2[-1] * temp[-2]
    else:
        # bottom layer # Dirichlet en bas:
        ab[1, -1] = 1.0
        ab[2, -2] = 0.0
        b[-1] = tbase

    # intermediate layers
    b[1:-1] = (timeterm[1:-1] - beta1[1:] - beta2[1:]) * \
        temp[1:-1] + beta1[1:] * temp[0:-2] + beta2[1:] * temp[2:]
    return ab, b


@jit((nfloat64[:], nfloat64, nfloat64, nfloat64, nfloat64[:], nfloat64[:], nfloat64[:], nfloat64[:], nfloat64), nopython=True, cache=True)
def compiled_cranknicolson_neuman_loop(temp, flux, dflux, dt, dx, ks, rho, cpice=2092.0, tbase=273):
    n = len(temp)
    ab = np.empty((3, n))
    b = np.empty(n)

    # a faire une seule fois
    dx_ks = dx / ks
    beta = 1 / (dx_ks[:-1] + dx_ks[1:])
    beta2 = beta

    # intermediate layers

    timeterm = (dx * rho * cpice) / dt

    # diagonal
    ab[1, 0] = timeterm[0] + beta2[0] - 0.5 * dflux  # top layer
    for i in range(1, n - 1):
        ab[1, i] = timeterm[i] + beta[i - 1] + beta2[i]

    for i in range(0, n - 1):
        # upperdiag
        ab[0, i + 1] = -beta2[i]
        # lowerdiag
        ab[2, i] = -beta2[i]

    if tbase is None:
        # bottom layer # flux nul
        ab[1, -1] = timeterm[-1] + beta2[-1]   # diag
        ab[2, -2] = -beta2[-1]  # lowerdiag
        # bottom layer # flux nul
        b[-1] = (timeterm[-1] - beta2[-1]) * temp[-1] + beta2[-1] * temp[-2]
    else:
        # bottom layer # Dirichlet en bas:
        ab[1, -1] = 1.0
        ab[2, -2] = 0.0
        b[-1] = tbase

    # right hand side
    b[0] = (timeterm[0] - beta2[0] - 0.5 * dflux) * temp[0] + beta2[0] * temp[1] + flux  # top layer
    # intermediate layers
    for i in range(1, n - 1):
        b[i] = (timeterm[i] - beta[i - 1] - beta2[i]) * temp[i] \
            + beta[i - 1] * temp[i - 1] \
            + beta2[i] * temp[i + 1]

    return ab, b



@jit((nfloat64[:], nfloat64, nfloat64, nfloat64, nfloat64[:], nfloat64[:], nfloat64[:], nfloat64[:], nfloat64), nopython=True, cache=True)
def compiled_cranknicolson_neuman_directsolver(temp, flux, dflux, dt, dx, ks, rho, cpice=2092.0, tbase=273):

    n = len(temp)
    diag = np.empty(n)
    b = np.empty(n)

    # a faire une seule fois
    dx_ks = dx / ks
    beta = 1 / (dx_ks[:-1] + dx_ks[1:])
    beta2 = beta

    timeterm = (dx * rho * cpice) / dt

    # diagonal
    diag[0] = timeterm[0] + beta2[0] - 0.5 * dflux  # top layer
    b[0] = (timeterm[0] - beta2[0] - 0.5 * dflux) * temp[0] + beta2[0] * temp[1] + flux  # top layer

    for i in range(1, n - 1):
        # lowerdiag is -beta2[i] except if directly at the base... put a 0
        m = (-beta2[i - 1]) / diag[i - 1]

        diag[i] = (timeterm[i] + beta[i - 1] + beta2[i]) \
            - m * (-beta[i - 1])  # tdma
        b[i] = (timeterm[i] - beta[i - 1] - beta2[i]) * temp[i] \
            + beta[i - 1] * temp[i - 1] \
            + beta2[i] * temp[i + 1] \
            - m * b[i - 1]  # tdma

    if tbase is None:
        m = (-beta2[n - 2]) / diag[n - 2]

        # bottom layer # flux nul
        diag[n - 1] = (timeterm[-1] + beta2[-1])  \
            - m * (-beta[n - 2])  # tdma

        # bottom layer # flux nul
        b[-1] = (timeterm[-1] - beta2[-1]) * temp[-1] + beta2[-1] * temp[-2] \
            - m * b[-2]  # tdma

    else:
        # m = 0 in the case of dirichlet because the lowerdiag is 0
        # bottom layer # Dirichlet en bas:
        diag[-1] = 1.0
        b[-1] = tbase

    # tdma back
    x = diag
    x[-1] = b[-1] / diag[-1]

    for il in range(n - 2, -1, -1):
        x[il] = (b[il] - (-beta[il]) * x[il + 1]) / diag[il]

    return x


# Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver

@jit((nfloat64[:], nfloat64[:], nfloat64[:], nfloat64[:]), nopython=True, cache=True)
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)

    Adapted to scipy solver convention on the lower/upper diag
    '''
    nf = len(d)  # number of equations

    # don't copy... destructive solver
    # bc, cc, dc = map(np.array, (b, c, d))  # copy arrays

    for it in range(1, nf):
        m = a[it - 1] / b[it - 1]
        b[it] -= m * c[it]
        d[it] -= m * d[it - 1]

    x = b
    x[-1] = d[-1] / b[-1]

    for il in range(nf - 2, -1, -1):
        x[il] = (d[il] - c[il + 1] * x[il + 1]) / b[il]

    return x
