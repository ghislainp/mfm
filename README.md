
Minimal Firn Model
==================

MFM is a simple model to compute the surface energy budget and temperature in the snowpack. It takes as input the meteorological forcing close to the surface (air temperature and humidity, wind speed, downwelling short wave and long wave radiation) and compute the surface temperature, the heat flux entering the snowpack and the propagation of the heat in the snowpack.

MFM is inspired from the surface energy budget model proposed by Essery and Etchevers, (2014), to which a thermal diffusion scheme has been added in order to compute the evolution of snow temperature profile. The model was originaly used in Picard et al. (2009, 2012) to model the Antarctic Firn. The code was in Fortran and was converted in 2013 in Python for teaching. The current version proposed here is in Python and has been used to investigate the thermal difference between the Arctic and Alpine snowpacks (Dominé et al. submitted).

To use MFM, you need to write a "driver" code which set the simulations parameters (snowpack layers and others), the meteorological forcing and call the "model" or "model_dirichlet" function from the mfm module. Examples of driver code in Jupyter notebook format are provided in the so called directory, including the code to produce the simulations used in Dominé et al. submitted.

Dependencies: MFM relies on numpy and scipy. However, it also uses numba to compile Python code and improve a bit the computation speed although it is possible to remove this dependency by editing the files if necessary. The example notebooks also need pandas and matplotlib to work. 

License information
--------------------

See the file ``LICENSE.txt`` for terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.

Copyright (c) 2013-2018 Ghislain Picard.


