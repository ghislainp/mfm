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


from numpy import empty, zeros
import scipy.linalg

from numba import jit
import numba


def cranknicolson_dirichlet(temp, tsurf, dt, dx, ks, rho, cpice=2092.0, tbase=None):
    ab, b = compiled_cranknicolson_dirichlet(temp, tsurf, dt, dx, ks, rho, cpice=cpice, tbase=tbase)
    return scipy.linalg.solve_banded((1, 1), ab, b)


@jit
def compiled_cranknicolson_dirichlet(temp, tsurf, dt, dx, ks, rho, cpice=2092.0, tbase=None):

    n = len(temp)
    ab = zeros((3, n))
    b = zeros(n)

    # a faire une seule fois
    dx_ks = dx / ks
    beta = empty(n)
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


def cranknicolson_neuman(temp, flux, dflux, dt, dx, ks, rho, cpice=2092.0, tbase=273):

    ab, b = compiled_cranknicolson_neuman(temp, flux, dflux, dt, dx, ks, rho, cpice, tbase)
    return scipy.linalg.solve_banded((1, 1), ab, b)

@jit((numba.float64[:], numba.float64, numba.float64, numba.int64, numba.float64[:], numba.float64[:], numba.float64[:], numba.float64[:], numba.float64), nopython=True, cache=True)
def compiled_cranknicolson_neuman(temp, flux, dflux, dt, dx, ks, rho, cpice=2092.0, tbase=273):

    n = len(temp)
    ab = zeros((3, n))
    b = zeros(n)

    # a faire une seule fois
    dx_ks = dx / ks
    beta = empty(n)
    beta[1:] = 1 / (dx_ks[:-1] + dx_ks[1:])
    beta[0] = 0

    beta1 = beta[:-1]
    beta2 = beta[1:]

    # intermediate layers

    timeterm = dx * rho * cpice / dt

    # diagonal
    ab[1, 0:-1] = timeterm[0:-1] + beta1 + beta2
    # upperdiag
    ab[0, 1:] = -beta2

    # lowerdiag
    # in principle it's beta1, but as lowerdiag is shifted by one step, shifted beta1 is beta2
    ab[2, :-1] = -beta2

    # top layer
    ab[1, 0] = timeterm[0] + beta2[0] - 0.5 * dflux  # diag
    ab[0, 1] = -beta2[0]  # upperdiag
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
