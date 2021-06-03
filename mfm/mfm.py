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


from collections import namedtuple
from .cranknicolson import cranknicolson_dirichlet, cranknicolson_neuman
from .surface_budget import surface_budget_essery

import numpy as np


# class data (python 3.7+ would be ideal for MeteoData and Parameters)
class MeteoData(object):

    def __init__(self, datetime=None, tair=None, qair=None, windspeed=None, swdn=None, lwdn=None, pressure=None):
        """meteorological variable for one time-step.

        :param datatime: date and time of the current time-step (optional).
        :param tair: air temperature (K).
        :param qair: specific humidity (kg/kg).
        :param windspeed: wind speed (m/s).
        :param swdn: downwelling short wave (W/m2).
        :param lwdn: downwelling long wave (W/m2).
        :param pressure: surface air pressure (Pa). Has a minimal impact, so a constant value is usually sufficient.
    """

        self.datetime = datetime
        self.tair = tair
        self.qair = qair
        self.windspeed = windspeed
        self.swdn = swdn
        self.lwdn = lwdn
        self.pressure = pressure


class Parameters(object):

    def __init__(self, albedo=None, z0=None, zt=None, dt=None, dx=None, rho=None, ks=None, cp=None, V=None, tbase=None):
        """parameters for the MFM model

        :param albedo: surface albedo (0-1).
        :param z0: aerodynamic roughness length (m).
        :param zt: height of air temperature measurements (m). It is often 2m.
        :param dt: time step (s). It is usually 60s - 3600s.
        :param dx: array with the tickness of every layer (m).
        :param rho: array with the density of every layer (kg/m3).
        :param ks: array with the thermal conductivity of every layer (Wm/K).
        :param cp: scalar or array with the thermal capacity of every layer (W/kg/°C).
        :param V: turbulent exchange in low wind conduction. See Essery and Etchevers, 2004 or Picard et al. 2009  
        """        

        self.albedo = albedo
        self.z0 = z0
        self.zt = zt
        self.dt = dt
        self.dx = dx 
        self.rho = rho
        self.ks = ks
        self.cp = cp
        self.V = V
        self.tbase = tbase


def depth(params, origin="surface"):
    """compute the depth array ("z") from the parameters

    :param params: parameters used to run the model
    :param origin: from where the depth is taken. "surface" or "soil". In any case "z" values are increasing upward.
    """

    z = -np.cumsum(np.insert(params.dx, 0, 0))[:-1]

    if origin == "soil":
        z -= z[-1]
    return z


def model(meteo, temp, params, return_flux=False, solver=None):
    """run MFM model for one time step. The computation includes solving the surface energy budget and the diffusion equation for one time step.None

    :param meteo: data with the near-surface meteorological conditions for the current time step.
    :param temp: a 1-d array with temperature in each layer at the previous time step.
    :param params: constant parameters needed by the model. See the documentation of the class Parameters.
    :param return_flux: whether to return the flux (in W/m2) or not.
    :param solver: type of solver to use. See crancknicholson.py for the available options

    :returns: temperature in each layer at the current time step and the surface flux if return_flux is True.
    """

    flux = surface_budget_essery(meteo.tair, meteo.qair, meteo.windspeed, meteo.pressure, temp[0],
                                 meteo.swdn, meteo.lwdn, params.albedo, params.z0, params.zt, params.V)
    deltaT = 0.1
    dflux = (surface_budget_essery(meteo.tair, meteo.qair, meteo.windspeed, meteo.pressure, temp[0] + deltaT,
                                   meteo.swdn, meteo.lwdn, params.albedo, params.z0, params.zt, params.V) - flux) / deltaT

    temp = cranknicolson_neuman(temp, flux, dflux, params.dt, params.dx, params.ks,
                                params.rho, cpice=params.cp, tbase=params.tbase, solver=solver)

    if return_flux:
        return temp, flux
    else:
        return temp


def model_dirichlet(ts, temp, params):
    """run the diffusion model for one time step. The computation takes the surface temperature to force the model temperature

    :param ts: surface temperature for the current time step.
    :param params: constant parameters needed by the model. See the documentation of the class Parameters.
    :param return_flux: whether to return the flux (in W/m2) or not.

    :returns: temperature in each layer at the current time step.
    """

    temp = cranknicolson_dirichlet(temp, ts, params.dt, params.dx, params.ks,
                                   params.rho, cpice=params.cp, tbase=params.tbase)
    return temp


