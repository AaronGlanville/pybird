#z1 NGC/SGC best fit:
#[3.08163075, 0.66270605, 0.11203498, 0.021665, 1.80328371, 0.54996484, 0.23224699, 1.82816374, 0.80374018, 0.22272056]

#z3 NGC/SGC best fit: 
#[2.63174571, 0.68489515, 0.12687036, 0.02165535, 2.36380242, 0.49054938,  0.08086074,  2.34623665,  0.71383503, -0.07937769]

import numpy as np
import sys
from configobj import ConfigObj
import scipy
from dynesty import plotting as dyplot
from dynesty import DynamicNestedSampler
import matplotlib
import matplotlib.pyplot as plt

sys.path.append("../")
from fitting_codes.fitting_utils_NGCSGC import (
    FittingData_NGC_SGC,
    BirdModel,
    create_plot_combined,
    update_plot_combined,
    update_plot_individual,
    format_pardict,
    do_optimization,
)

def lnpost(params, birdmodel, fittingdata, plt):

    # This returns the posterior distribution which is given by the log prior plus the log likelihood
    prior = lnprior(params, birdmodel)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike(params, birdmodel, fittingdata, plt)
    return prior + like

def prior_transform(u, birdmodel):
    x = np.array(u)

    lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
    upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta

    #flat prior on ln10As:
        
    x[0] = (u[0] * (upper_bounds[0] - lower_bounds[0])) + lower_bounds[0]
        
    #flat prior on h

    x[1] = (u[1] * (upper_bounds[1] - lower_bounds[1])) + lower_bounds[1]

    #flat prior on Omega_cdm

    x[2] = (u[2] * (upper_bounds[2] - lower_bounds[2])) + lower_bounds[2]

    #omega_b prior from BBN
    mu, sigma = 0.02235, 0.00049
    x[3] = scipy.stats.norm.ppf(u[3], loc=mu, scale=sigma) # (gaussian centered on 0.02235, width of 0.00049)

    if birdmodel.pardict["do_marg"]:
    #indexed NGC z1, SGC z1, NGC z3, SGC z3
    #b1_NGC
        #x[4] = u[4] * 3.0 #b1_NGC prior transform, originally from 0-3
        mu, sigma = 2.487, 0.345
        x[4] = scipy.stats.norm.ppf(u[4], loc=mu, scale=sigma) #  (gaussian on b1 from single data fits)
    #c2_NGC
        x[5] = (u[5] - 0.5) * 8.0 #c2_NGC, range from -4 to 4
    #c4_NGC
        mu, sigma = 0, 2
        x[6] = scipy.stats.norm.ppf(u[6], loc=mu, scale=sigma) #c4_NGC, Gaussian centered around 0 with width of 2
    #b1_SGC
        #x[7] = u[7] * 3 #b1_SGC prior transform
        mu, sigma = 2.572, 0.535
        x[7] = scipy.stats.norm.ppf(u[7], loc=mu, scale=sigma) # (gaussian on b1 from single data fits)
    #c2_SGC
        x[8] = (u[8] - 0.5) * 8.0 #c2_NGC, range from -4 to 4
    #c4_SGC
        mu, sigma = 0, 2
        x[9] = scipy.stats.norm.ppf(u[9], loc=mu, scale=sigma) #c4_NGC, Gaussian centered around 0 with width of 2

   
    else:
    #b1_NGC:
        x[4] = u[4] * 3.0 #b1_NGC, range from 0 to 3
    #c2_NGC:
        x[5] = (u[5] - 0.5) * 8.0 #c2_NGC, range from -4 to 4
    #b3_NGC:
        mu, sigma = 0, 2
        x[6] = scipy.stats.norm.ppf(u[6], loc=mu, scale=sigma) #b3_NGC, gaussian of mu = 0, sigma = 2
    #c4_NGC:
        mu, sigma = 0, 2
        x[7] = scipy.stats.norm.ppf(u[7], loc=mu, scale=sigma) #c4_NGC, gaussian of mu = 0, sigma = 2
    #cct_NGC:
        mu, sigma = 0, 2
        x[8] = scipy.stats.norm.ppf(u[8], loc=mu, scale=sigma) #cct_NGC, gaussian of mu = 0, sigma = 2
    #cr1_NGC:
        mu, sigma = 0, 4
        x[9] = scipy.stats.norm.ppf(u[9], loc=mu, scale=sigma) #cr1_NGC, gaussian of mu = 0, sigma = 4
    #cr2_NGC:
        mu, sigma = 0, 4
        x[10] = scipy.stats.norm.ppf(u[10], loc=mu, scale=sigma) #cr2_NGC, gaussian of mu 0, sigma = 4
    #ce1_NGC:
        mu, sigma = 0, 2
        x[11] = scipy.stats.norm.ppf(u[11], loc=mu, scale=sigma) #ce1_NGC, gaussian of mu = 0, sigma = 2
    #cemono_NGC:
        mu, sigma = 0, 2
        x[12] = scipy.stats.norm.ppf(u[12], loc=mu, scale=sigma)  #cemono_NGC, gaussian of mu = 0, sigma = 2
    #cequad_NGC:
        mu, sigma = 0, 2
        x[13] = scipy.stats.norm.ppf(u[13], loc=mu, scale=sigma) #cequad_NGC, gaussian of mu = 0, sigma = 2
    #bnlo_NGC:
        mu, sigma = 0, 2
        x[14] = scipy.stats.norm.ppf(u[14], loc=mu, scale=sigma) #bnlo_NGC, gaussian of mu = 0, sigma = 2
#----------------------------------------------------------------------------------------------------------------
    #b1_SGC:
        x[15] = u[15] * 3.0 #b1_SGC, range from 0 to 3
    #c2_SGC:
        x[16] = (u[16] - 0.5) * 8.0 #c2_SGC, range from -4 to 4
    #b3_SGC:
        mu, sigma = 0, 2
        x[17] = scipy.stats.norm.ppf(u[17], loc=mu, scale=sigma) #b3_SGC, gaussian of mu = 0, sigma = 2
    #c4_SGC:
        mu, sigma = 0, 2
        x[18] = scipy.stats.norm.ppf(u[18], loc=mu, scale=sigma) #c4_SGC, gaussian of mu = 0, sigma = 2
    #cct_SGC:
        mu, sigma = 0, 2
        x[19] = scipy.stats.norm.ppf(u[19], loc=mu, scale=sigma) #cct_SGC, gaussian of mu = 0, sigma = 2
    #cr1_SGC:
        mu, sigma = 0, 4
        x[20] = scipy.stats.norm.ppf(u[20], loc=mu, scale=sigma) #cr1_SGC, gaussian of mu = 0, sigma = 4
    #cr2_SGC:
        mu, sigma = 0, 4
        x[21] = scipy.stats.norm.ppf(u[21], loc=mu, scale=sigma) #cr2_SGC, gaussian of mu 0, sigma = 4
    #ce1_SGC:
        mu, sigma = 0, 2
        x[22] = scipy.stats.norm.ppf(u[22], loc=mu, scale=sigma) #ce1_SGC, gaussian of mu = 0, sigma = 2
    #cemono_SGC:
        mu, sigma = 0, 2
        x[23] = scipy.stats.norm.ppf(u[23], loc=mu, scale=sigma)  #cemono_SGC, gaussian of mu = 0, sigma = 2
    #cequad_SGC:
        mu, sigma = 0, 2
        x[24] = scipy.stats.norm.ppf(u[24], loc=mu, scale=sigma) #cequad_SGC, gaussian of mu = 0, sigma = 2
    #bnlo_SGC:
        mu, sigma = 0, 2
        x[25] = scipy.stats.norm.ppf(u[25], loc=mu, scale=sigma) #bnlo_SGC, gaussian of mu = 0, sigma = 2
    return x

##def lnprior(params, birdmodel):
#
#    # Here we define the prior for all the parameters. We'll ignore the constants as they
#    # cancel out when subtracting the log posteriors
#    if birdmodel.pardict["do_marg"]:
#        b1_NGC, c2_NGC, c4_NGC = params[4:7] #in order [cosmo], [b_NGC], [b_SGC]
#        b1_SGC, c2_SGC, c4_SGC = params[7:10]
#    else:
#        b1_NGC, c2_NGC, b3_NGC, c4_NGC, cct_NGC, cr1_NGC, cr2_NGC, ce1_NGC, cemono_NGC, cequad_NGC, bnlo_NGC = params[4:15]
#        b1_SGC, c2_SGC, b3_SGC, c4_SGC, cct_SGC, cr1_SGC, cr2_SGC, ce1_SGC, cemono_SGC, cequad_SGC, bnlo_SGC = params[15:26]
#
#    ln10As, h, omega_cdm, omega_b = params[:4]
#
#    lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
#    upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta
#
#    # Flat priors for cosmological parameters
#    if np.any(np.less([ln10As, h, omega_cdm, omega_b], lower_bounds)) or np.any(
#        np.greater([ln10As, h, omega_cdm, omega_b], upper_bounds)
#    ):
#        return -np.inf
#
#    # BBN (D/H) inspired prior on omega_b
#    omega_b_prior = -0.5 * (omega_b - 0.02166) ** 2 / 0.00037 ** 2
#
#    # Flat prior for b1
#    if b1_NGC < 0.0 or b1_NGC > 3.0:
#        return -np.inf
#    if b1_SGC < 0.0 or b1_SGC > 3.0:
#        return -np.inf
#
#    # Flat prior for c2
#    if c2_NGC < -4.0 or c2_NGC > 4.0:
#        return -np.inf
#    if c2_SGC < -4.0 or c2_SGC > 4.0:
#        return -np.inf
#
#    # Gaussian prior for c4
#    c4_prior_NGC = -0.5 * 0.25 * c4_NGC ** 2
#    c4_prior_SGC = -0.5 * 0.25 * c4_SGC ** 2
#
#    if birdmodel.pardict["do_marg"]:
#
#        return omega_b_prior + c4_prior_NGC + c4_prior_SGC
#
#    else:
#        # Gaussian prior for b3 of width 2 centred on 0
#        b3_prior_NGC = -0.5 * 0.25 * b3_NGC ** 2
#
#        # Gaussian prior for cct of width 2 centred on 0
#        cct_prior_NGC = -0.5 * 0.25 * cct_NGC ** 2
#
#        # Gaussian prior for cr1 of width 4 centred on 0
#        cr1_prior_NGC = -0.5 * 0.0625 * cr1_NGC ** 2
#
#        # Gaussian prior for cr1 of width 4 centred on 0
#        cr2_prior_NGC = -0.5 * 0.0625 * cr2_NGC ** 2
#
#        # Gaussian prior for ce1 of width 2 centred on 0
#        ce1_prior_NGC = -0.5 * 0.25 * ce1_NGC ** 2
#
#        # Gaussian prior for cemono of width 2 centred on 0
#        cemono_prior_NGC = -0.5 * 0.25 * cemono_NGC ** 2
#
#        # Gaussian prior for cequad of width 2 centred on 0
#        cequad_prior_NGC = -0.5 * 0.25 * cequad_NGC ** 2
#
#        # Gaussian prior for bnlo of width 2 centred on 0
#        bnlo_prior_NGC = -0.5 * 0.25 * bnlo_NGC ** 2
#---------------------------------------------------------------
#        # Gaussian prior for b3 of width 2 centred on 0
#        b3_prior_SGC = -0.5 * 0.25 * b3_SGC ** 2
#
#        # Gaussian prior for cct of width 2 centred on 0
#        cct_prior_SGC = -0.5 * 0.25 * cct_SGC ** 2
#
#        # Gaussian prior for cr1 of width 4 centred on 0
#        cr1_prior_SGC = -0.5 * 0.0625 * cr1_SGC ** 2
#
#        # Gaussian prior for cr1 of width 4 centred on 0
#        cr2_prior_SGC = -0.5 * 0.0625 * cr2_SGC ** 2
#
#        # Gaussian prior for ce1 of width 2 centred on 0
#        ce1_prior_SGC = -0.5 * 0.25 * ce1_SGC ** 2
#
#        # Gaussian prior for cemono of width 2 centred on 0
#        cemono_prior_SGC = -0.5 * 0.25 * cemono_SGC ** 2
#
#        # Gaussian prior for cequad of width 2 centred on 0
#        cequad_prior_SGC = -0.5 * 0.25 * cequad_SGC ** 2
#
#        # Gaussian prior for bnlo of width 2 centred on 0
#        bnlo_prior_SGC = -0.5 * 0.25 * bnlo_SGC ** 2
#
#        return (
#            omega_b_prior
#            + c4_prior_NGC
#            + b3_prior_NGC
#            + cct_prior_NGC
#            + cr1_prior_NGC
#            + cr2_prior_NGC
#            + ce1_prior_NGC
#            + cemono_prior_NGC
#            + cequad_prior_NGC
#            + bnlo_prior_NGC
#
#            + c4_prior_SGC
#            + b3_prior_SGC
#            + cct_prior_SGC
#            + cr1_prior_SGC
#            + cr2_prior_SGC
#            + ce1_prior_SGC
#            + cemono_prior_SGC
#            + cequad_prior_SGC
#            + bnlo_prior_SGC
#        )


def lnlike(params, birdmodel, fittingdata, plt):

    if birdmodel.pardict["do_marg"]:
        b2_SGC = (params[-2] + params[-1]) / np.sqrt(2.0)
        b4_SGC = (params[-2] - params[-1]) / np.sqrt(2.0)
        bs_SGC = [params[-3], b2_SGC, 0.0, b4_SGC, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        b2_NGC = (params[-5] + params[-4]) / np.sqrt(2.0)
        b4_NGC = (params[-5] - params[-4]) / np.sqrt(2.0)
        bs_NGC = [params[-6], b2_NGC, 0.0, b4_NGC, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    else:
        b2_SGC = (params[-10] + params[-8]) / np.sqrt(2.0)
        b4_SGC = (params[-10] - params[-8]) / np.sqrt(2.0)
        bs_SGC = [
            params[-11],
            b2_SGC,
            params[-9],
            b4_SGC,
            params[-7],
            params[-6],
            params[-5],
            params[-4] * SGC_shot_noise,
            params[-3] * SGC_shot_noise,
            params[-2] * SGC_shot_noise,
            params[-1],
        ]

        b2_NGC = (params[-21] + params[-19]) / np.sqrt(2.0)
        b4_NGC = (params[-21] - params[-19]) / np.sqrt(2.0)
        bs_NGC = [
            params[-22],
            b2_NGC,
            params[-20],
            b4_NGC,
            params[-18],
            params[-17],
            params[-16],
            params[-15] * NGC_shot_noise,
            params[-14] * NGC_shot_noise,
            params[-13] * NGC_shot_noise,
            params[-12],
        ]

    # Get the bird model
    ln10As, h, omega_cdm, omega_b = params[:4]

    Plin_NGC, Ploop_NGC, Plin_SGC, Ploop_SGC = birdmodel.compute_pk_separategrids([ln10As, h, omega_cdm, omega_b])

    P_model, P_model_interp, P_model_NGC, P_model_interp_NGC, P_model_SGC, P_model_interp_SGC = birdmodel.compute_model_separategrids(bs_NGC, bs_SGC, Plin_NGC, Ploop_NGC, Plin_SGC, Ploop_SGC, fittingdata.data["x_data"])
    
    #returns both individual NGC/SGC Pi, and joint (full matrix w/ zeros)

    Pi_NGC, Pi_SGC, Pi = birdmodel.get_Pi_for_marg_separategrids(Ploop_NGC, Ploop_SGC, bs_NGC[0], bs_SGC[0], NGC_shot_noise, SGC_shot_noise, fittingdata.data["x_data"])
    chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data) #CHECK THIS IN SOME DETAIL- should we be using joint Pi?
    return -0.5 * chi_squared
    #print(chi_squared)
'''
    if plt is not None:
        chi_squared_print = chi_squared
        #if birdmodel_combined.pardict["do_marg"]:
        if birdmodel.pardict["do_marg"]:
            bs_analytic = birdmodel.compute_bestfit_analytic(Pi, fittingdata.data, P_model_interp) #Should output NGC-SGC bias params in one object
            #print("bs_analytic")
            #print(bs_analytic)
            birdmodel_combined.pardict["do_marg"] = 0
            #pardict_NGC["do_marg"] = 0
            
            b2_SGC = (params[-2] + params[-1]) / np.sqrt(2.0)
            b4_SGC = (params[-2] - params[-1]) / np.sqrt(2.0)

            b2_NGC = (params[-5] + params[-4]) / np.sqrt(2.0)
            b4_NGC = (params[-5] - params[-4]) / np.sqrt(2.0)
            #MAKE SURE TO CHECK- 
            bs_NGC = [
                params[-6],
                b2_NGC,
                bs_analytic[0],
                b4_NGC,
                bs_analytic[1],
                bs_analytic[2],
                bs_analytic[3],
                bs_analytic[4] * NGC_shot_noise,
                bs_analytic[5] * NGC_shot_noise,
                bs_analytic[6] * NGC_shot_noise,
                bs_analytic[7],
            ]

            bs_SGC = [
                params[-3],
                b2_SGC,
                bs_analytic[8],
                b4_SGC,
                bs_analytic[9],
                bs_analytic[10],
                bs_analytic[11],
                bs_analytic[12] * SGC_shot_noise,
                bs_analytic[13] * SGC_shot_noise,
                bs_analytic[14] * SGC_shot_noise,
                bs_analytic[15],
            ]
            P_model, P_model_interp, P_model_NGC, P_model_interp_NGC, P_model_SGC, P_model_interp_SGC = birdmodel.compute_model_separategrids(bs_NGC, bs_SGC, Plin_NGC, Ploop_NGC, Plin_SGC, Ploop_SGC, fittingdata_combined.data["x_data"])
            #print("P_model_interp")
            #print(P_model_interp)
            chi_squared_print = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata_combined.data)
            #birdmodel_combined.pardict["do_marg"] = 1
            birdmodel.pardict["do_marg"] = 1
        update_plot_combined(pardict_NGC, fittingdata.data["x_data"], P_model_interp_NGC, P_model_interp_SGC, colours, "NGC+SGC_z1_combined", plt)
        #update_plot_individual(pardict_NGC, fittingdata.data["x_data"], P_model_interp_NGC, plt)
        #update_plot_individual(pardict_NGC, fittingdata.data["x_data"], P_model_interp_SGC, plt)
'''


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    #NGC_configfile = "../config/tbird_NGC_z1_s10fixed_singlefit.txt"
    #SGC_configfile = "../config/tbird_SGC_z1_s10fixed_singlefit.txt"
    #NGC_configfile = "../config/tbird_NGC_z3_s10fixed_singlefit.txt"
    #SGC_configfile = "../config/tbird_SGC_z3_s10fixed_singlefit.txt"
    #NGC_configfile = "../config/tbird_NGC_z1_s10fixed_combined_fits_nbodykit.txt"
    #SGC_configfile = "../config/tbird_SGC_z1_s10fixed_combined_fits_nbodykit.txt"
    NGC_configfile = "../config/tbird_NGC_z3_s10fixed_combined_fits_nbodykit.txt"
    SGC_configfile = "../config/tbird_SGC_z3_s10fixed_combined_fits_nbodykit.txt"
    plot_flag = int(sys.argv[1])
    pardict_NGC = ConfigObj(NGC_configfile)
    pardict_SGC = ConfigObj(SGC_configfile)
    print(pardict_NGC)
    print(pardict_SGC)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict_NGC = format_pardict(pardict_NGC)
    pardict_SGC = format_pardict(pardict_SGC)
    NGC_shot_noise = float(pardict_NGC["shot_noise"])
    SGC_shot_noise = float(pardict_SGC["shot_noise"])
    print("NGC/SGC shot noise = %lf %lf"%(NGC_shot_noise, SGC_shot_noise))

    # Set up the data
    fittingdata_combined = FittingData_NGC_SGC(pardict_NGC, pardict_SGC, NGC_shot_noise, SGC_shot_noise)
    print("fittingdata_NGC")
    print(fittingdata_combined.data["fit_data_NGC"])
    print("fittingdata_SGC")
    print(fittingdata_combined.data["fit_data_SGC"])

    # Set up the BirdModel
    birdmodel_combined = BirdModel(pardict_NGC, template=False)

    # Plotting (for checking/debugging, should turn off for production runs)
    #plt = None
    #if plot_flag:
    #    colours = ["b", "g", "r", "c", "n"]
    #    plt = create_plot_combined(pardict_NGC, pardict_SGC, colours, fittingdata_combined) #NOTE: Pardict NGC must contain both gridnames for the NGC and SGC sections

    if birdmodel_combined.pardict["do_marg"]:
        start = np.concatenate([birdmodel_combined.valueref[:4], [1.3, 0.5, 0.5], [1.3, 0.5, 0.5]])
    else:
        start = np.concatenate([birdmodel_combined.valueref[:4], [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

    # Does an optimization
    #result = do_optimization(lambda *args: -lnpost(*args), start, birdmodel_combined, fittingdata_combined, plt)

    # Does an MCMC
    #do_emcee(lnpost, start, birdmodel_combined, fittingdata_combined, plt)

    # Calculates evidences using Dynesty
    if birdmodel_combined.pardict["do_marg"]:
        ndim = 10
    else:
        ndim = 26
    
    #log_like = lnlike(start, birdmodel, fittingdata, plt)
    dsampler = DynamicNestedSampler(lnlike, prior_transform, ndim, logl_args=[birdmodel_combined, fittingdata_combined, plt], ptform_args=[birdmodel_combined]) #Loglikelihood function, prior transform, and the number of dimensions taken by log likelihood
    dsampler.run_nested() #just to speed up testing for saving outputs
    dres = dsampler.results
    print(dres)
    samples = dres["samples"]
    print(len(samples))
    log_weights = dres["logwt"]
    weights = np.exp(dres["logwt"] - dres["logz"][-1])
    max_weight = weights.max()
    trimmed_samples = []
    trimmed_weights = []
    size = len(samples)
    for i in range(size):
        if weights[i] > (max_weight/1e5):
            trimmed_samples.append(samples[i, :])
            trimmed_weights.append(weights[i])
    print(len(trimmed_weights))
    output = np.column_stack((trimmed_samples, trimmed_weights))
    print(output)
    np.savetxt("dynesty_NGC_SGC_z3_nbodykit_om_2p235_err_49_Gaussian_b1_prior.dat", output)

    #plt.cla()
    #plt.clf()
    #plt.close()
    #fig, ax = plt.subplots(figsize=(10,8))
    #plt.axhline(y=0.1)
    #plt.show()
    fig, axes = dyplot.traceplot(dres, trace_cmap='viridis', connect=True)
    
    plt.savefig("dynesty_NGC_SGC_z3_nbodykit_om_2p235_err_49_Gaussian_b1_prior.pdf")
