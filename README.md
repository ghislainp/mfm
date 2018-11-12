
Minimal Firn Model
==================

MFM (Minimal Firn Model) is a simple model to compute the surface energy budget and temperature in the snowpack. It takes as input the meteorological forcing close to the surface (air temperature and humidity, wind speed, downwelling short wave and long wave radiation) and computes the surface temperature, the heat flux entering the snowpack and the propagation of heat in the snowpack.

MFM is inspired from the surface energy budget model proposed by Essery and Etchevers, (2014), to which a thermal diffusion scheme has been added in order to compute the evolution of snow temperature profile. The model was originaly used in Picard et al. (2009, 2012) to model the Antarctic Firn. The code was initially in Fortran and was converted in 2013 in Python for teaching. The current version proposed here is in Python and has been used to investigate the thermal difference between Arctic and Alpine snowpacks (Domine et al. submitted).

To use MFM, you need to write a "driver" code which set the simulations parameters (snowpack layers and others), the meteorological forcing and call the "model" or "model_dirichlet" function from the mfm module. Examples of driver code in Jupyter notebook format are provided in the so called directory, including the code to produce the simulations used in Domine et al. (submitted).

Dependencies: MFM relies on numpy and scipy. However, it also uses numba to compile Python code and improve a bit the computation speed although it is possible to remove this dependency by editing the files if necessary. The example notebooks also need pandas and matplotlib to work.


References:
------------

* R. Essery and P. Etchevers, Parameter sensitivity in simulations of snowmelt, Journal of Geophysical Research - Atmosphere, 109, D20111, 2004, doi:10.1029/2004JD005036

* G. Picard, L. Brucker, M. Fily, H. Gallée and G. Krinner, Modeling timeseries of microwave brightness temperature in Antarctica. Journal of Glaciology, 55 (191), pp 537-551, 2009, doi:10.3189/002214309788816678

* G. Picard, F. Domine, G. Krinner, L. Arnaud & E. Lefebvre, Inhibition of the positive snow-albedo feedback by precipitation in interior Antarctica, Nature Climate Change, 2, 795–798 2012, doi:10.1038/nclimate1590

* F. Domine, G. Picard, S. Morin, M. Barrere, J.-B. Madore, A. Langlois, Major Issues in Simulating some Arctic Snowpack Properties Using Current Detailed Snow Physics Models. Consequences for the Thermal Regime and Water Budget of Permafrost, submitted to Journal of Advances in Modeling Earth Systems.


License information
--------------------

See the file ``LICENSE.txt`` for terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.

Copyright (c) 2013-2018 Ghislain Picard.


