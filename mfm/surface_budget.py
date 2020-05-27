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
from numba import jit, float64


# goff-gracht over water below0 C
# http://cires.colorado.edu/~voemel/vp.html
@jit((float64,), nopython=True, cache=True)
def vaporsaturation_liquid(T):

    log10ew = -7.90298 * (373.16 / T - 1) + 5.02808 * np.log10(373.16 / T) \
        - 1.3816e-7 * (10**(11.344 * (1 - T / 373.16)) - 1) \
        + 8.1328e-3 * (10**(-3.49149 * (373.16 / T - 1)) - 1) \
        + np.log10(1013.246)

    return 100 * 10**log10ew


# Goff Gratch equation over ice
# http://cires.colorado.edu/~voemel/vp.html
@jit((float64,), nopython=True, cache=True)
def vaporsaturation_ice(T):

    log10ei = -9.09718 * (273.16 / T - 1) \
        - 3.56654 * np.log10(273.16 / T) \
        + 0.876793 * (1 - T / 273.16) \
        + np.log10(6.1071)

    return 100 * 10**log10ei


@jit((float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64), nopython=True, cache=True)
def surface_budget_essery(tair, qair, windspeed, pressure, ts, swdn, lwdn, albedo, z0, zt, V=0):

    # constant
    # fusion heat (J/kg)
    # lambdaf = 3.335e5

    # sublimation heat (J/kg)
    lambdas = 2.838e6
    # air density (kg/m3)
    rhoair = pressure / (287.0 * tair)
    # air specific heat capacity [J/KG-K]
    cpair = 1005.0
    # boltzmann constant [W/m^2-K^4]
    sigma = 5.669E-8
    # snow emissivity in the thermal infrared
    #epsilon = 1.00

    # humidity
    eps = 0.622
    # psatair = vaporsaturation_liquid(tair)
    # qair = rh * psatair / (pressure - (1 - eps) * rh * psatair)

    psatsurf = vaporsaturation_ice(ts)
    qsatsurf = eps * psatsurf / (pressure - (1 - eps) * psatsurf)

    # compute net shortwave
    net_shortwave = (1 - albedo) * swdn

    # compute net longwave
    net_longwave = lwdn - sigma * ts**4

    # turbulent fluxes
    von_karman = 0.4
    chn = von_karman**2 / (np.log(zt / z0))**2

    # conductance
    # bulk richardson number
    if windspeed > 0:
        rib = 9.81 * zt / windspeed**2 * ((tair - ts) / tair + (qair - qsatsurf) / (qair + eps / (1 - eps)))

        # compute the correction function
        if rib >= 0:
            # stable condition
            fn = 1 / (1 + 10 * rib)
        else:
            # unstable condition
            fz = 0.25 * np.sqrt(z0 / zt)
            fn = 1 - 10 * rib / (1 + 10 * chn * np.sqrt(-rib) / fz)
    else:
        fn = 1

    ga = chn * fn * windspeed

    # sensible flux
    sensible = (rhoair * cpair * ga + V) * (tair - ts)

    # latent flux
    latent = lambdas * (rhoair * ga +  V/cpair) * (qair - qsatsurf)

    # bilan complet
    G = net_shortwave + net_longwave + sensible + latent

    return G


