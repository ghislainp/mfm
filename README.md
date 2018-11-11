
Minimal Firn Model
==================

MFM is a simple model to compute the surface energy budget and temperature in the snow. It takes as input the meteorological forcing close to the surface (air temperature and humidity, wind speed, downwelling short wave and long wave radiation) and compute the surface temperature, the heat flux entering the snowpack and the propagatation of the heat in the snowpack.

MFM is inspired from the surface energy budget model proposed by Essery and Etchevers, 2014, to which a thermal diffusion scheme has been added to compute snow temperature evolution. The model was used in Picard et al. 2009 to model the Antarctic Firn. It 

MFM relies on no more than numpy and scipy. However, it also uses numba to compile Python code and improve a bit the computation speed. It is possible to remove this dependency by editing the files.

License information
--------------------

See the file ``LICENSE.txt`` for terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.

Copyright (c) 2013-2018 Ghislain Picard.


