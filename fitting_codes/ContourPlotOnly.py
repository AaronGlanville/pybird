# -*- coding: utf-8 -*-
"""
File to create contour plot of simultaneous fits
"""

import numpy as np
import h5py
import emcee
import corner
from configobj import ConfigObj
import matplotlib.pyplot as plt
import sys
sys.path.append("../")

file = "z1_NGC+NGC_40000_Stept_Test.hdf5"
reader = emcee.backends.HDFBackend(file)
samples = reader.get_chain(flat=True)
corner.corner(samples)

