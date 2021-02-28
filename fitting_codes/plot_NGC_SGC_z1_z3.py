#Code to plot NGC/SGC z1+z3 optimizations to check
#the quality of fits across all datasets

#currently have an optimised vector of 
#x: array([ 2.89199760e+00,  6.72635335e-01,  1.17520433e-01,  2.16563139e-02,
#        1.98488062e+00,  7.66091829e-01,  5.11518623e-03,  1.99997093e+00,
#        1.02666324e+00,  1.05747221e-01,  2.05534685e+00,  7.90368280e-02,
#        2.16201773e+00,  2.01396846e+00, -2.18477291e-02, -5.12116608e+00])

# cosmo params: [2.89199760e+00 6.72635335e-01 1.17520433e-01 2.16563139e-02]
# NGC z1 bias params: [1.98488062e+00,  7.66091829e-01,  5.11518623e-03]  
# SGC z1 bias params: [1.99997093e+00, 1.02666324e+00,  1.05747221e-01]
# NGC z3 bias params: [2.05534685e+00, 7.90368280e-02, 2.16201773e+00]
# SGC z3 bias params: [2.01396846e+00, -2.18477291e-02, -5.12116608e+00]

#z1 NGC/SGC only best fit:
# cosmo params: [3.08163075, 0.66270605, 0.11203498, 0.021665]
# NGC z1 bias params: [1.80328371, 0.54996484, 0.23224699]
# SGC z1 bias params: [1.82816374, 0.80374018, 0.22272056]

#these SGC z1 bias params give decent fit:
#chi_2 = 266.828535
#cosmo = 2.956978, 0.670104, 0.115952, 0.021654
#SGC z1 bias params = 1.901623 0.888681 0.058302

#these SGC z1 bias params give bad fit (too low on monopole):
#chi_2 = 267.051412
#cosmo = 2.883820, 0.678147, 0.122655, 0.021964
#SGC z1 bias params = 1.986489 0.771001 0.715260

#After applying very strict priors- still dipping down heavily
#chi_2 = 278.419699
#cosmo = 3.015344, 0.683491, 0.133635, 0.021976
#SGC z1 bias params = 1.715480 0.735149 0.054584

import numpy as np
import sys
from configobj import ConfigObj

sys.path.append("../")
from fitting_codes.fitting_utils_NGCSGC import (
    FittingData_NGC_SGC,
    BirdModel,
    create_plot_combined,
    create_plot_NGC_SGC_z1_z3,
    update_plot_combined,
    update_plot_NGC_SGC_z1_z3,
    update_plot_individual,
    format_pardict,
    do_optimization,
)

#NGC+SGC z1/z3
params = [2.89199760e+00,  6.72635335e-01,  1.17520433e-01,  2.16563139e-02, 1.98488062e+00,  7.66091829e-01,  5.11518623e-03,  1.99997093e+00, 1.02666324e+00,  1.05747221e-01,  2.05534685e+00,  7.90368280e-02, 2.16201773e+00,  2.01396846e+00, -2.18477291e-02, -5.12116608e+00]

if __name__ == "__main__":

    NGC_z1_configfile = "../config/tbird_NGC_z1_s10fixed_singlefit.txt"
    SGC_z1_configfile = "../config/tbird_SGC_z1_s10fixed_singlefit.txt"
    NGC_z3_configfile = "../config/tbird_NGC_z3_s10fixed_singlefit.txt"
    SGC_z3_configfile = "../config/tbird_SGC_z3_s10fixed_singlefit.txt"
    pardict_z1_NGC = ConfigObj(NGC_z1_configfile)
    pardict_z1_SGC = ConfigObj(SGC_z1_configfile)
    pardict_z3_NGC = ConfigObj(NGC_z3_configfile)
    pardict_z3_SGC = ConfigObj(SGC_z3_configfile)

    pardict_z1_NGC = format_pardict(pardict_z1_NGC)
    pardict_z1_SGC = format_pardict(pardict_z1_SGC)
    pardict_z3_NGC = format_pardict(pardict_z3_NGC)
    pardict_z3_SGC = format_pardict(pardict_z3_SGC)

    NGC_z1_shot_noise = float(pardict_z1_NGC["shot_noise"])
    SGC_z1_shot_noise = float(pardict_z1_SGC["shot_noise"])
    NGC_z3_shot_noise = float(pardict_z3_NGC["shot_noise"])
    SGC_z3_shot_noise = float(pardict_z3_SGC["shot_noise"])

    fittingdata_z1_combined = FittingData_NGC_SGC(pardict_z1_NGC, pardict_z1_SGC, NGC_z1_shot_noise, SGC_z1_shot_noise)
    fittingdata_z3_combined = FittingData_NGC_SGC(pardict_z3_NGC, pardict_z3_SGC, NGC_z3_shot_noise, SGC_z3_shot_noise)

    # Set up the BirdModel
    #Only need to pass single pardict when creating the birdmodel
    birdmodel_z1_combined = BirdModel(pardict_z1_NGC, template=False)
    birdmodel_z3_combined = BirdModel(pardict_z3_NGC, template=False)

    colours = ["b", "g", "r", "c", "m", "y", "k", "mediumorchid"]
    plt = create_plot_NGC_SGC_z1_z3(pardict_z1_NGC, pardict_z1_SGC, pardict_z3_NGC, pardict_z3_SGC, colours, fittingdata_z1_combined, fittingdata_z3_combined) #NOTE: Pardict NGC must contain both gridnames for the NGC and SGC sections

    #-------------------------------------------------------------------------------------
    #With the models/data instantiated, and the plot object constructed, we now must 
    #construct the full data vector from our marginalised output

    b2_NGC_z1 = (params[5] + params[6])/np.sqrt(2.0) #CHECK INDEXING
    b4_NGC_z1 = (params[5] - params[6])/np.sqrt(2.0)
    bs_NGC_z1 = [params[4], b2_NGC_z1, 0, b4_NGC_z1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    b2_SGC_z1 = (params[8] + params[9])/np.sqrt(2.0)
    b4_SGC_z1 = (params[8] - params[9])/np.sqrt(2.0)
    bs_SGC_z1 = [params[7], b2_SGC_z1, 0, b4_SGC_z1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    b2_NGC_z3 = (params[11] + params[12])/np.sqrt(2.0) #CHECK INDEXING
    b4_NGC_z3 = (params[11] - params[12])/np.sqrt(2.0)
    bs_NGC_z3 = [params[10], b2_NGC_z3, 0, b4_NGC_z3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    b2_SGC_z3 = (params[14] + params[15])/np.sqrt(2.0)
    b4_SGC_z3 = (params[14] - params[15])/np.sqrt(2.0)
    bs_SGC_z3 = [params[13], b2_SGC_z3, 0, b4_SGC_z3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    Plin_NGC_z1, Ploop_NGC_z1, Plin_SGC_z1, Ploop_SGC_z1 = birdmodel_z1_combined.compute_pk_separategrids(params[:4])
    Plin_NGC_z3, Ploop_NGC_z3, Plin_SGC_z3, Ploop_SGC_z3 = birdmodel_z3_combined.compute_pk_separategrids(params[:4])

    P_model_z1, P_model_interp_z1, P_model_NGC_z1, P_model_interp_NGC_z1, P_model_SGC_z1, P_model_interp_SGC_z1 = birdmodel_z1_combined.compute_model_separategrids(bs_NGC_z1, bs_SGC_z1, Plin_NGC_z1, Ploop_NGC_z1, Plin_SGC_z1, Ploop_SGC_z1, fittingdata_z1_combined.data["x_data"])
    P_model_z3, P_model_interp_z3, P_model_NGC_z3, P_model_interp_NGC_z3, P_model_SGC_z3, P_model_interp_SGC_z3 = birdmodel_z3_combined.compute_model_separategrids(bs_NGC_z3, bs_SGC_z3, Plin_NGC_z3, Ploop_NGC_z3, Plin_SGC_z3, Ploop_SGC_z3, fittingdata_z3_combined.data["x_data"])

    Pi_NGC_z1, Pi_SGC_z1, Pi_z1 = birdmodel_z1_combined.get_Pi_for_marg_separategrids(Ploop_NGC_z1, Ploop_SGC_z1, bs_NGC_z1[0], bs_SGC_z1[0], fittingdata_z1_combined.data["shot_noise_NGC"], fittingdata_z1_combined.data["shot_noise_SGC"], fittingdata_z1_combined.data["x_data"])
    Pi_NGC_z3, Pi_SGC_z3, Pi_z3 = birdmodel_z3_combined.get_Pi_for_marg_separategrids(Ploop_NGC_z3, Ploop_SGC_z3, bs_NGC_z3[0], bs_SGC_z3[0], fittingdata_z3_combined.data["shot_noise_NGC"], fittingdata_z3_combined.data["shot_noise_SGC"], fittingdata_z3_combined.data["x_data"])

    bs_analytic_z1 = birdmodel_z1_combined.compute_bestfit_analytic(Pi_z1, fittingdata_z1_combined.data, P_model_interp_z1) #Should output NGC-SGC bias params in one object
    bs_analytic_z3 = birdmodel_z3_combined.compute_bestfit_analytic(Pi_z3, fittingdata_z3_combined.data, P_model_interp_z3) 

    bs_NGC_z1 = [
        params[4],
        b2_NGC_z1,
        bs_analytic_z1[0],
        b4_NGC_z1,
        bs_analytic_z1[1],
        bs_analytic_z1[2],
        bs_analytic_z1[3],
        bs_analytic_z1[4] * fittingdata_z1_combined.data["shot_noise_NGC"],
        bs_analytic_z1[5] * fittingdata_z1_combined.data["shot_noise_NGC"],
        bs_analytic_z1[6] * fittingdata_z1_combined.data["shot_noise_NGC"],
        bs_analytic_z1[7],
    ]

    bs_SGC_z1 = [
        params[7],
        b2_SGC_z1,
        bs_analytic_z1[8],
        b4_SGC_z1,
        bs_analytic_z1[9],
        bs_analytic_z1[10],
        bs_analytic_z1[11],
        bs_analytic_z1[12] * fittingdata_z1_combined.data["shot_noise_SGC"],
        bs_analytic_z1[13] * fittingdata_z1_combined.data["shot_noise_SGC"],
        bs_analytic_z1[14] * fittingdata_z1_combined.data["shot_noise_SGC"],
        bs_analytic_z1[15],
    ]

            #----------------------------------------------------

    bs_NGC_z3 = [
        params[10],
        b2_NGC_z3,
        bs_analytic_z3[0],
        b4_NGC_z3,
        bs_analytic_z3[1],
        bs_analytic_z3[2],
        bs_analytic_z3[3],
        bs_analytic_z3[4] * fittingdata_z3_combined.data["shot_noise_NGC"],
        bs_analytic_z3[5] * fittingdata_z3_combined.data["shot_noise_NGC"],
        bs_analytic_z3[6] * fittingdata_z3_combined.data["shot_noise_NGC"],
        bs_analytic_z3[7],
    ]

    b2_SGC_z3 = (params[14] + params[15])/np.sqrt(2.0)
    b4_SGC_z3 = (params[14] - params[15])/np.sqrt(2.0)

    bs_SGC_z3 = [
        params[13],
        b2_SGC_z3,
        bs_analytic_z3[8],
        b4_SGC_z3,
        bs_analytic_z3[9],
        bs_analytic_z3[10],
        bs_analytic_z3[11],
        bs_analytic_z3[12] * fittingdata_z3_combined.data["shot_noise_SGC"],
        bs_analytic_z3[13] * fittingdata_z3_combined.data["shot_noise_SGC"],
        bs_analytic_z3[14] * fittingdata_z3_combined.data["shot_noise_SGC"],
        bs_analytic_z3[15],
    ]

    P_model_z1, P_model_interp_z1, P_model_NGC_z1, P_model_interp_NGC_z1, P_model_SGC_z1, P_model_interp_SGC_z1 = birdmodel_z1_combined.compute_model_separategrids(bs_NGC_z1, bs_SGC_z1, Plin_NGC_z1, Ploop_NGC_z1, Plin_SGC_z1, Ploop_SGC_z1, fittingdata_z1_combined.data["x_data"])
    P_model_z3, P_model_interp_z3, P_model_NGC_z3, P_model_interp_NGC_z3, P_model_SGC_z3, P_model_interp_SGC_z3 = birdmodel_z3_combined.compute_model_separategrids(bs_NGC_z3, bs_SGC_z3, Plin_NGC_z3, Ploop_NGC_z3, Plin_SGC_z3, Ploop_SGC_z3, fittingdata_z3_combined.data["x_data"])

    update_plot_NGC_SGC_z1_z3(pardict_z1_NGC, pardict_z3_NGC, fittingdata_z1_combined.data["x_data"], fittingdata_z3_combined.data["x_data"], P_model_interp_NGC_z1, P_model_interp_SGC_z1, P_model_interp_NGC_z3, P_model_interp_SGC_z3, colours, "output_NGC_SGC_z1_z3.png", plt, keep=True)
