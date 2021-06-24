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

#First run:
num_runs = 90
points_per_run = 50

static_sample_list = []

for i in range(num_runs):
    print(i)
    static_sample = np.load("static_chains/z1_z2_z3_curved_fits_true_data_kmax_0p2_0p2_0p2_knl_0p4_10x_prior_window/100x50_z1_z2_z3_kmax_0p20_0p20_0p20_knl_0p5_curved_true_data_10x_eft_prior_width_"+str(i)+".npy", allow_pickle=True)
    static_sample_list.append(static_sample)

#print("reading static samples")
#static_sample = np.load("chains/Static_Chains/dres_NGC_SGC_z3_static_sample_nlive_500_flat.npy", allow_pickle=True)
#print("static samples")
#print(static_sample)

static_sample_loaded_list = []
for i in range(num_runs):
    static_sample = static_sample_list[i]
    static_sample_loaded = static_sample[()]
    static_sample_loaded_list.append(static_sample_loaded)

#test = static_sample[()]
#print(test)

#print(static_sample_loaded_list[1])
merged = dyfunc.merge_runs(static_sample_loaded_list)

#merged = dyfunc.merge_runs([test, test, test, test, test])
#merged = dyfunc.merge_runs([test])
#print("Merged!!")

#-----------------Test dynesty output from merged chain------------

#The raw output gives log weights- we correct this, and then trim our 
#sample to remove extreme outliers

samples = merged.samples
log_weights = merged.logwt
log_z = merged.logz
log_l = merged.logl

#------------------Now we solved for evidence/KL divergence------

samples = np.stack(samples, axis=0)
log_weights = np.stack(log_weights, axis=0)
shape = len(log_weights)
log_weights = log_weights.reshape(shape, 1)
log_z = np.stack(log_z, axis=0)
log_z = log_z.reshape(shape, 1)
log_l = np.stack(log_l, axis=0)
log_l = log_l.reshape(shape, 1)

anesthetic_sample = np.concatenate((samples, log_l, log_z, log_weights,), axis=1)
#columns = ['ln10As', 'h', 'Omega_cdm', 'Omega_b', 'b1', 'c2', 'c4', 'logL', 'logZ', 'logWeights']

#Single redshift bin:
#columns = ['ln10As', 'h', 'Omega_cdm', 'Omega_b', 'b1 NGC', 'c2 NGC', 'b1 SGC', 'c2 SGC', 'logL', 'logZ', 'logWeights']

#3 redshift bins:
columns = ['ln10As', 'h', 'Omega_cdm', 'Omega_b', 'Omega_k', 'b1 NGC z1', 'c2 NGC z1', 'b1 SGC z1', 'c2 SGC z1', 'b1 NGC z2', 'c2 NGC z2', 'b1 SGC z2', 'c2 SGC z2', 'b1 NGC z3', 'c2 NGC z3', 'b1 SGC z3', 'c2 SGC z3', 'logL', 'logZ', 'logWeights']

nested= NestedSamples(data=anesthetic_sample, columns=columns)

nested.nlive= num_runs * points_per_run

ns_output = nested.ns_output()
for x in ns_output:
    print('%10s = %9.2f +/- %4.2f' % (x, ns_output[x].mean(), ns_output[x].std()))
    
#------------------And finally, we plot the posterior------------

weights = np.exp(log_weights - log_z[-1])
max_weight = weights.max()
trimmed_samples = []
trimmed_weights = []
size = len(samples)
for i in range(size):
    if weights[i] > (max_weight/1e5):
        trimmed_samples.append(samples[i, :])
        trimmed_weights.append(weights[i])
print(len(trimmed_weights))
trimmed_samples = np.stack(trimmed_samples, axis=0)
trimmed_weights = np.stack(trimmed_weights, axis=0)

Omega_m = (trimmed_samples[:, 2] + trimmed_samples[: ,3])/(trimmed_samples[:, 1]**2) #Check formatting
print(Omega_m)

c = ChainConsumer()
c.add_chain(np.hstack([trimmed_samples[:, :5], Omega_m[:, None]]), parameters=["$ln10^{10} A_s$", "$h$", "$\Omega_{\rm{cdm}}$", "$\Omega_b$", "$\Omega_k$", "$\Omega_m$"], name = "knl=0.4, kmax=0.2,0.2,0.2, widened EFT priors", weights=trimmed_weights)

#------------------Now, we repeat for our second sample of points------

#Second run:
num_runs = 29
points_per_run = 50

static_sample_list = []

for i in range(num_runs):
    print(i)
    static_sample = np.load("static_chains/z1_z2_z3_curved_fits_true_data_kmax_0p2_0p2_0p2_knl_0p5/100x50_z1_z2_z3_kmax_0p20_0p20_0p20_knl_0p5_curved_true_data_"+str(i)+".npy", allow_pickle=True)
    static_sample_list.append(static_sample)

#print("reading static samples")
#static_sample = np.load("chains/Static_Chains/dres_NGC_SGC_z3_static_sample_nlive_500_flat.npy", allow_pickle=True)
#print("static samples")
#print(static_sample)

static_sample_loaded_list = []
for i in range(num_runs):
    static_sample = static_sample_list[i]
    static_sample_loaded = static_sample[()]
    static_sample_loaded_list.append(static_sample_loaded)

#test = static_sample[()]
#print(test)

#print(static_sample_loaded_list[1])
merged = dyfunc.merge_runs(static_sample_loaded_list)

#merged = dyfunc.merge_runs([test, test, test, test, test])
#merged = dyfunc.merge_runs([test])
#print("Merged!!")

#-----------------Test dynesty output from merged chain------------

#The raw output gives log weights- we correct this, and then trim our 
#sample to remove extreme outliers

samples = merged.samples
log_weights = merged.logwt
log_z = merged.logz
log_l = merged.logl

#------------------Now we solved for evidence/KL divergence------

samples = np.stack(samples, axis=0)
log_weights = np.stack(log_weights, axis=0)
shape = len(log_weights)
log_weights = log_weights.reshape(shape, 1)
log_z = np.stack(log_z, axis=0)
log_z = log_z.reshape(shape, 1)
log_l = np.stack(log_l, axis=0)
log_l = log_l.reshape(shape, 1)

anesthetic_sample = np.concatenate((samples, log_l, log_z, log_weights,), axis=1)
#columns = ['ln10As', 'h', 'Omega_cdm', 'Omega_b', 'b1', 'c2', 'c4', 'logL', 'logZ', 'logWeights']

#Single redshift bin:
#columns = ['ln10As', 'h', 'Omega_cdm', 'Omega_b', 'b1 NGC', 'c2 NGC', 'c4 NGC', 'b1 SGC', 'c2 SGC', 'c4 SGC', 'logL', 'logZ', 'logWeights']

#3 redshift bins:
columns = ['ln10As', 'h', 'Omega_cdm', 'Omega_b', 'Omega_k', 'b1 NGC z1', 'c2 NGC z1', 'b1 SGC z1', 'c2 SGC z1', 'b1 NGC z2', 'c2 NGC z2', 'b1 SGC z2', 'c2 SGC z2', 'b1 NGC z3', 'c2 NGC z3', 'b1 SGC z3', 'c2 SGC z3', 'logL', 'logZ', 'logWeights']

nested= NestedSamples(data=anesthetic_sample, columns=columns)

nested.nlive= num_runs * points_per_run

ns_output = nested.ns_output()
for x in ns_output:
    print('%10s = %9.2f +/- %4.2f' % (x, ns_output[x].mean(), ns_output[x].std()))
    
#------------------And finally, we plot the posterior------------

weights = np.exp(log_weights - log_z[-1])
max_weight = weights.max()
trimmed_samples = []
trimmed_weights = []
size = len(samples)
for i in range(size):
    if weights[i] > (max_weight/1e5):
        trimmed_samples.append(samples[i, :])
        trimmed_weights.append(weights[i])
print(len(trimmed_weights))
trimmed_samples = np.stack(trimmed_samples, axis=0)
trimmed_weights = np.stack(trimmed_weights, axis=0)

Omega_m = (trimmed_samples[:, 2] + trimmed_samples[: ,3])/(trimmed_samples[:, 1]**2) #Check formatting
print(Omega_m)

c.add_chain(np.hstack([trimmed_samples[:, :5], Omega_m[:, None]]), parameters=["$ln10^{10} A_s$", "$h$", "$\Omega_{\rm{cdm}}$", "$\Omega_b$", "$\Omega_k$", "$\Omega_m$"], name = "knl=0.5, kmax=0.2,0.2,0.2, D'Amico EFT priors", weights=trimmed_weights)

#------------------Finally, we plot both chains-----------------------

fig = c.plotter.plot(filename="comparing_merged_runs.pdf", display=False) #Without truths
#fig = c.plotter.plot(truth={"$h$": 0.6777, "$\Omega_{\rm{cdm}}$":0.119, "$\Omega_b$":0.02204, "$\Omega_m$":0.307115, "$\Omega_k$":0.0}, filename="z1_z2_z3_100x100_curved_kmax_0p2_Bianchi_mocks.pdf", display=False) #Without truths

summary = c.analysis.get_summary()
print(summary)

