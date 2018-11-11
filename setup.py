from distutils.core import setup, Command
from setuptools import find_packages

setup(
    name = "mfm",
    packages = find_packages(exclude='test'),
    version = "1.0",
    description = "Minimal Firn Model",
    author = "Ghislain Picard",
    author_email = "ghislain.picard@univ-grenoble-alpes.fr",
    url = "https://github.com/ghislainp/mfm",
    keywords = ["snow","model","surface energy budget","thermal diffusion", "temperature"],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Public License (GPL)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        ],
    long_description = """\
The Minimal Firn Model (MFM) is inspired from teh Minimal Snow Model imagined by R. Essey (U. Edinbourgh). It computes the surface energy budget and 
thermal diffusion in a snowpack made of fixed layers. It is mainly used to predict surface and internal temperature of constant or slowly changing snowpacks.
"""
)
