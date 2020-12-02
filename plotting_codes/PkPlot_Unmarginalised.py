# -*- coding: utf-8 -*-
"""
Code to plot last Pk of sampler (as a test)
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

#final_state = [3.21603996, 0.66177365, 0.11406556, 0.02166129, 1.72148287, 0.31766473, 0.16708188, 1.72329827, 0.31783722, 0.27889942]
#final_state = [3.14378963, 0.662368  , 0.11062639, 0.02165865, 1.82274245, 0.38287651, 0.3448505 , 1.80906487, 0.50814912, 0.10566116]
#final_state = [2.35536815,  0.6646729,  0.11085787,  0.02164821,  2.92905179,-4,  0.19146507, -0.501881, -0.22990709,  3.70252655, -0.38528887,  0.00913787,  0.07956661, -0.80266268,  0.32542347,2.87218632, -3.05988198,  1.13824445,  0.88798916,  0.05223853,1.22546558, -1.19839734,  0.03229475, -0.0142107 , -0.09042414, 4.23870186]
final_state = [ 2.35336665e+00,  6.64365675e-01,  1.10832085e-01,  2.16487202e-02, 2.94197166e+00, -3.99999996e+00,  3.48171654e-01, -1.58742192e-01, -7.22422534e-01, -5.54977094e-01,  3.69171274e+00, -2.56679754e-02, 2.42315426e-03, -4.19686325e-01, -3.93437437e-01,  2.89468033e+00, -2.87882979e+00, -3.10056603e-02, -7.15358460e-01,  1.37735719e-01, -7.01538683e-01,  1.72636551e+00, -2.81313357e-02, -3.33598629e-03, -5.46540176e-01,  2.42616626e+00]

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
        
b2_NGC = (final_state[-21] + final_state[-19]) / np.sqrt(2.0)
b4_NGC = (final_state[-21] - final_state[-19]) / np.sqrt(2.0)
bs_NGC = [
            final_state[-22],
            b2_NGC,
            final_state[-20],
            b4_NGC,
            final_state[-18],
            final_state[-17],
            final_state[-16],
            final_state[-15] * NGC_shot_noise,
            final_state[-14] * NGC_shot_noise,
            final_state[-13] * NGC_shot_noise,
            final_state[-12],
            ]

b2_SGC = (final_state[-10] + final_state[-8]) / np.sqrt(2.0)
b4_SGC = (final_state[-10] - final_state[-8]) / np.sqrt(2.0)
bs_SGC = [
            final_state[-11],
            b2_SGC,
            final_state[-9],
            b4_SGC,
            final_state[-7],
            final_state[-6],
            final_state[-5],
            final_state[-4] * SGC_shot_noise,
            final_state[-3] * SGC_shot_noise,
            final_state[-2] * SGC_shot_noise,
            final_state[-1],
            ]
Plin, Ploop = birdmodel_combined.compute_pk(final_state[:4]) #Plin/Ploop only relies on cosmology, does not need NGC/SGC distinction
P_model, P_model_interp = birdmodel_combined.compute_model_NGCSGC(bs_NGC, bs_SGC, Plin, Ploop, fittingdata_combined.data["x_data"])#just need k-space, should be the same across files
print("P_model")
print(P_model_interp)
print(fittingdata_combined.data["x_data"])

Pi = birdmodel_combined.get_Pi_for_marg_NGCSGC(Ploop, bs_NGC[0], bs_SGC[0], NGC_shot_noise, SGC_shot_noise, fittingdata_combined.data["x_data"]) #Pi seems kinda dodgy- interpolated P from discrete points? idk    
print("Pi[1]= ")
NGC_SGC_k = np.hstack((fittingdata_combined.data["x_data"][0], fittingdata_combined.data["x_data"][1],fittingdata_combined.data["x_data"][0], fittingdata_combined.data["x_data"][1]))
print(NGC_SGC_k)
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
    plt_err = np.concatenate((x_data, x_data)).flatten() ** 1.5 * np.sqrt(cov[np.diag_indices(nx0 + nx2 + nx0 + nx2)])
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
        color="black",
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
        color="yellow",
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
plt.savefig("Unmarg_BOSSNGC_z1NGC+SGC_CombinedFit_WinFunc_s10fixed_40000Steps.png")
plt.show()

