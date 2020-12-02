# -*- coding: utf-8 -*-
"""
Code to plot model output of Nelder-mead optimisation for analytically marginalised parameters

Simply update the final state defined [4 cosmo params, 3 NGC bias, 3 SGC bias], and 
the code will evaluate the best fitting linear order bias parameters for each NGC/SGC,
before passing these and plotting the full model
"""

import numpy as np
import h5py
import emcee
import corner
from configobj import ConfigObj
import matplotlib.pyplot as plt
import sys
sys.path.append("../")

from fitting_codes.fitting_utils_NGCSGC_combined import (
    FittingData_NGCSGC,
    BirdModel,
    create_plot_combined,
    update_plot,
    format_pardict,
    do_optimization,
)

final_state = [3.15295823, 0.66150917, 0.10982741, 0.02165857, 1.80434092, 0.30575538, 0.27275935, 1.82378376, 0.56648497, 0.13806955]

configfile_BOSS_NGC_z1 = "../config/tbird_NGC_z1.txt"
configfile_BOSS_SGC_z1 = "../config/tbird_SGC_z1.txt" #For joing NGC/SGC z1 fits
pardict_BOSS_NGC_z1 = ConfigObj(configfile_BOSS_NGC_z1)
print(pardict_BOSS_NGC_z1)
pardict_BOSS_SGC_z1 = ConfigObj(configfile_BOSS_SGC_z1)
n_datasets = 2 #Used when assigning walkers in emcee

#LIST EVERY INDIVIDUAL PARDICT

# Just converts strings in pardicts to numbers in int/float etc.
pardict_BOSS_NGC_z1 = format_pardict(pardict_BOSS_NGC_z1)
pardict_BOSS_SGC_z1 = format_pardict(pardict_BOSS_SGC_z1)
    
pardict_NGC = pardict_BOSS_NGC_z1
pardict_SGC = pardict_BOSS_SGC_z1

NGC_shot_noise = float(pardict_BOSS_NGC_z1["shot_noise"])
SGC_shot_noise = float(pardict_BOSS_SGC_z1["shot_noise"])
print("NGC/SGC shot noise = %lf %lf"%(NGC_shot_noise, SGC_shot_noise))

pardicts = [pardict_BOSS_NGC_z1, pardict_BOSS_SGC_z1]

#FORMAT EVERY INDIVIDUAL INI

# Set up the data
fittingdata_combined = FittingData_NGCSGC(pardict_BOSS_NGC_z1, pardict_BOSS_SGC_z1, NGC_shot_noise, SGC_shot_noise)

#Check to make sure everything is defined correctly:
print("fittingdata_BOSS_NGC_z1 = ")
print(pardict_BOSS_NGC_z1)
print("fittingdata_Combined = ")
#CREATE LIST OF FITTING DATA

fittingdata = fittingdata_combined    

if pardict_NGC["do_hex"]:
    x_data = fittingdata_combined.data["x_data"]
    nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), len(x_data[2])
    print("nx0 = %lf" %nx0)
else:
    print(fittingdata.data["x_data"][:2])
    x_data = fittingdata_combined.data["x_data"][:2]
    nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), 0

# Set up the BirdModel
birdmodel_combined = BirdModel(pardict_BOSS_NGC_z1, template=False) #just sets up the EFT model, does not need to be duplicated for NGC/SGC cuts
print(fittingdata_combined.data["fit_data_NGC"])

b2_NGC = (final_state[5] + final_state[6])/np.sqrt(2.0)
b4_NGC = (final_state[5] - final_state[6])/np.sqrt(2.0)
        
b2_SGC = (final_state[8] + final_state[9]) / np.sqrt(2.0)
b4_SGC = (final_state[8] - final_state[9]) / np.sqrt(2.0) #check indexing if any issues come up

#Testing to see whether plotting model with 11 linear bias  parameters analytically marginalised give reasonable plots

Plin, Ploop = birdmodel_combined.compute_pk(final_state[:4]) #Plin/Ploop only relies on cosmology, does not need NGC/SGC distinction
Pi = birdmodel_combined.get_Pi_for_marg_NGCSGC(Ploop, final_state[-6], final_state[-3], NGC_shot_noise, SGC_shot_noise, fittingdata_combined.data["x_data"]) #Pi seems kinda dodgy- interpolated P from discrete points? idk    
print("Pi[1]= ")

bs_analytic = birdmodel_combined.compute_bestfit_analytic(Pi, fittingdata_combined.data)
bs_NGC_analytic = [
            final_state[-6],
            b2_NGC,
            bs_analytic[0],
            b4_NGC,
            bs_analytic[1],
            bs_analytic[2],
            bs_analytic[3],
            bs_analytic[4],
            bs_analytic[5],
            bs_analytic[6],
            bs_analytic[7],
       ]

print("bs_NGC_analytic = ")
print(bs_NGC_analytic)

bs_SGC_analytic = [
            final_state[-3],
            b2_SGC,
            bs_analytic[8],
            b4_SGC,
            bs_analytic[9],
            bs_analytic[10],
            bs_analytic[11],
            bs_analytic[12],
            bs_analytic[13],
            bs_analytic[14],
            bs_analytic[15],
       ] 

print("bs_NSC_analytic = ")
print(bs_SGC_analytic)


P_model, P_model_interp = birdmodel_combined.compute_model_NGCSGC(bs_NGC_analytic, bs_SGC_analytic, Plin, Ploop, fittingdata_combined.data["x_data"])#just need k-space, should be the same across files
print("P_model")
print(P_model_interp)
print(fittingdata_combined.data["x_data"])

NGC_SGC_k = np.hstack((fittingdata_combined.data["x_data"][0], fittingdata_combined.data["x_data"][1],fittingdata_combined.data["x_data"][0], fittingdata_combined.data["x_data"][1]))
print(NGC_SGC_k)
P_model_interp = P_model_interp*np.sqrt(NGC_SGC_k)
#-------------------------------------------------------------------------------
#Plotting code:

fit_data_NGC = fittingdata.data["fit_data_NGC"]
fit_data_SGC = fittingdata.data["fit_data_SGC"]
cov = fittingdata.data["cov"]

plt_data_NGC = (
    np.concatenate(x_data) ** 2 * fit_data_NGC if pardict_NGC["do_corr"] else (np.concatenate(x_data)**1.5) * fit_data_NGC 
)
print("plt_data_NGC = ")
print(plt_data_NGC)
plt_data_SGC = (
    np.concatenate(x_data) ** 2 * fit_data_SGC if pardict_SGC["do_corr"] else (np.concatenate(x_data)**1.5) * fit_data_SGC 
)
print("plt_data_SGC = ")
print(plt_data_SGC)
if pardict_NGC["do_corr"]:
    plt_err = np.concatenate(x_data) ** 2 * np.sqrt(cov[np.diag_indices(nx0 + nx2 + nx4)])
else:
    plt_err = np.concatenate(x_data) ** 1.5 * np.sqrt(cov[np.diag_indices(nx0 + nx2)])
    print("plt_err = ")
    print(plt_err)

#NGC Monopole + Quadrupole
plt.errorbar(
    x_data[0],
    plt_data_NGC[:nx0],
    yerr=plt_err[:nx0],
    marker="o",
    markerfacecolor="r",
    markeredgecolor="k",
    color="r",
    linestyle="None",
    markeredgewidth=1.3,
    zorder=5,
)
plt.errorbar(
        x_data[1],
        plt_data_NGC[nx0 : nx0 + nx2],
        yerr=plt_err[nx0 : nx0 + nx2],
        marker="o",
        markerfacecolor="b",
        markeredgecolor="k",
        color="b",
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )
    
#SGC Monopole + Quadrupole (just reusing NGC errors, but oh well) 
plt.errorbar(
        x_data[0],
        plt_data_SGC[:nx0],
        yerr=plt_err[:nx0],
        marker="o",
        markerfacecolor="black",
        markeredgecolor="k",
        color="r",
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )
plt.errorbar(
        x_data[1],
        plt_data_SGC[nx0 : nx0 + nx2],
        yerr=plt_err[nx0 : nx0 + nx2],
        marker="o",
        markerfacecolor="yellow",
        markeredgecolor="k",
        color="b",
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )


for i in range(len(x_data[0])): #NGC P0 only as test
    P_model_interp[i] = P_model_interp[i]*((x_data[0][i])**1.5)
    P_model_interp[i+len(x_data[0])] = P_model_interp[i+len(x_data[0])]*((x_data[1][i]**1.5))
    P_model_interp[i+(2*len(x_data[0]))] = P_model_interp[i+(2*len(x_data[0]))]*((x_data[0][i]**1.5))
    P_model_interp[i+(3*len(x_data[0]))] = P_model_interp[i+(3*len(x_data[0]))]*((x_data[1][i]**1.5))

plt.plot(x_data[0], P_model_interp[:33], color="r")
plt.plot(x_data[1], P_model_interp[33:66], color="b")
plt.plot(x_data[0], P_model_interp[66:99], color="black")
plt.plot(x_data[1], P_model_interp[99:], color="yellow")
print("P_model_interp = ")
print(P_model_interp[:33])

plt.xlim(0.0, np.amax(pardict_NGC["xfit_max"]) * 1.05)
if pardict_NGC["do_corr"]:
   plt.xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$", fontsize=16)
   plt.ylabel(r"$s^{2}\xi(s)$", fontsize=16, labelpad=5)
else:
        plt.xlabel(r"$k\,(h\,\mathrm{Mpc}^{-1})$", fontsize=16)
        plt.ylabel(r"$k^{3/2}P(k)\,(h^{-3/2}\,\mathrm{Mpc^{3/2}})$", fontsize=16, labelpad=5)
plt.tick_params(width=1.3)
plt.tick_params("both", length=10, which="major")
plt.tick_params("both", length=5, which="minor")
for axis in ["top", "left", "bottom", "right"]:
        plt.gca().spines[axis].set_linewidth(1.3)
for tick in plt.gca().xaxis.get_ticklabels():
        tick.set_fontsize(14)
for tick in plt.gca().yaxis.get_ticklabels():
        tick.set_fontsize(14)
plt.tight_layout()
plt.gca().set_autoscale_on(False)
plt.ion()
plt.savefig("Marg_BOSSNGC_z1NGC+SGC_CombinedFit_WinFunc_s10fixed_40000Steps.png")
plt.show()


