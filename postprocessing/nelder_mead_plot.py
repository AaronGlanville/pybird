# -*- coding: utf-8 -*-
"""
Code to try and stick two dynesty runs together (to see if we can trivially parallelise
our static nested sampling runs, running an array with fewer live points, then gluing together)
"""

import requests
import tarfile
from anesthetic import MCMCSamples, NestedSamples, make_1d_axes
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import dynesty
from dynesty import utils as dyfunc
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import cmasher as cmr

num_runs = 1000

static_sample_list = []

c = ChainConsumer()
#3 redshift bins, no c4
columns = ['ln10As', 'h', 'Omega_cdm', 'Omega_b', 'Omega_k', 'b1 NGC z1', 'c2 NGC z1', 'c4 NGC z1', 'b1 SGC z1', 'c2 SGC z1', 'c4 SGC z1', 'b1 NGC z2', 'c2 NGC z2', 'c4 NGC z2', 'b1 SGC z2', 'c2 SGC z2', 'c4 SGC z2', 'b1 NGC z3', 'c2 NGC z3', 'c4 NGC z3', 'b1 SGC z3', 'c2 SGC z3', 'c4 SGC z3', 'logL', 'logZ', 'logWeights', 'lnpost']

static_sample_merged = []

for i in range(num_runs):
    print(i)
    #static_sample = np.loadtxt("basin_hopping_fits/basin_hopping_z1_z2_z3_p4_kmax_0p1_knl_0p8/basin_hopping_z1_z2_z3_individual_output_P4_kmax_0p1_testing_de_parameters_knl_1p5_"+str(i+1)+".txt")
    static_sample = np.loadtxt("basin_hopping_fits/basin_hopping_z1_z2_z3_p4_kmax_0p1_knl_0p8_corrected_redshifts/basin_hopping_z1_z2_z3_individual_output_P4_kmax_0p1_knl_0p8_corrected_redshifts_"+str(i+1)+".txt")
    #static_sample = np.loadtxt("basin_hopping_fits/basin_hopping_z1_z2_z3_p4_kmax_0p1_knl_1p5/basin_hopping_z1_z2_z3_individual_output_P4_kmax_0p1_testing_de_parameters_knl_1p5_"+str(i+1)+".txt")
    #static_sample = np.loadtxt("basin_hopping_fits/basin_hopping_z1_z2_z3_p4_kmax_0p1_knl_1p0/basin_hopping_z1_z2_z3_individual_output_P4_kmax_0p1_testing_de_parameters_knl_1p0_"+str(i+1)+".txt")
    #static_sample = np.loadtxt("basin_hopping_fits/basin_hopping_z1_z2_z3_testing_sampler_params/basin_hopping_z1_z2_z3_individual_output_P4_kmax_0p1_testing_de_parameters_knl_0p8_"+str(i+1)+".txt")
    static_sample_merged.append(static_sample)
    static_sample_list.append(static_sample)
    
static_sample_merged = np.concatenate(static_sample_list, axis=0).reshape(num_runs,-1)
print(static_sample_merged)

parameters = ["$ln10^{10} A_s$", "$h$", "$\Omega_{cdm} h^2$", "$\Omega_{b} h^2$"]#, "$\Omega_k$"]

loglike = static_sample_merged[:,-1]
logmin, logmax = np.min(loglike), np.max(loglike)
color_norm = []

#assign integer from 1 to 10 for colours based on log-likelihood

for i in range(len(loglike)):
    norm_log_like = 100 - np.int((((loglike[i] - logmin)/(logmax-logmin))*90) + 1) #+1 to guarantee no zero terms, chainconsumer doesn't like that
    color_norm.append(norm_log_like)

color = cmr.take_cmap_colors('Blues', 100, return_fmt='hex')

c = ChainConsumer()
for i in range(num_runs):
    print(i)
    #c.add_marker((np.hstack([static_sample_merged[i, :3], static_sample_merged[i,4]])), marker_size=5, parameters=parameters, color=color[color_norm[i]], name="knl = 0.8, corrected redshifts, kmax=0.10")
    c.add_marker((np.hstack([static_sample_merged[i, :4]])), marker_size=5, parameters=parameters, color="b", name="z1+z2+z3 mock mean")

c.configure(color_params="$logWeights$")

extents = [(2.3, 4.1), (0.6, 0.75), (0.09, 0.135), (0.02195, 0.02215)]

#fig = c.plotter.plot(truth={"$ln10^{10} A_s$":3.08, "$h$": 0.6777, "$\Omega_{cdm} h^2$":0.119, "$\Omega_b$":0.02204, "$\Omega_m$":0.307115, "$\Omega_k$":0.0}, filename="basin_hopping_z1_z2_z3_individual_output_P4_kmax_0p1_padded_edge.pdf", display=False) #Without truths
fig = c.plotter.plot(truth={"$ln10^{10} A_s$":3.08,"$h$": 0.6777, "$\Omega_{cdm} h^2$":0.119, "$\Omega_b$":0.02204, "$\Omega_m$":0.307115, "$\Omega_k$":0.0}, filename="basin_hopping_z1_z2_z3_individual_output.pdf", display=False, extents=extents) #Without truths

#c.add_marker(np.hstack([static_sample_merged[i, :5]]), parameters=parameters)
#plt.scatter(static_sample_merged[:, 0], static_sample_merged[:, 4])
    

