#Code to simultaneously fit NGC/SGC patches 
#across multiple redshift bins

import numpy as np
import scipy
import sys
from configobj import ConfigObj
from dynesty import plotting as dyplot
from dynesty import DynamicNestedSampler
import matplotlib
import matplotlib.pyplot as plt

sys.path.append("../")
from fitting_codes.fitting_utils_NGCSGC import (
    FittingData_NGC_SGC,
    BirdModel,
    #create_plot_combined,
    create_plot_NGC_SGC_z1_z3,
    #update_plot_combined,
    update_plot_NGC_SGC_z1_z3,
    #update_plot_individual,
    format_pardict,
    do_optimization,
)

def do_emcee(func, start, birdmodel, fittingdata, plt):

    import emcee
    for i, (pardict, model, data) in enumerate(zip(pardicts, birdmodel, fittingdata)):
        print("model:")
        print(model) 
        nparams = len(start)
        nwalkers = nparams*2
        print("nparams")
        print(nparams)
        print("nwalkers")
        print(nwalkers)
        print("start")
        begin = [[(0.01 * (np.random.rand() - 0.5) + 1.0) * start[j] for j in range(len(start))] for i in range(nwalkers)]
        print(begin)

    #marg_str = "marg" if pardict["do_marg"] else "all"
    #hex_str = "hex" if pardict["do_hex"] else "nohex"
    #dat_str = "xi" if pardict["do_corr"] else "pk"
    #fmt_str = "%s_%s_%2dhex%2d_%s_%s_%s.hdf5" if pardict["do_corr"] else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s.hdf5"
    #fitlim = birdmodel.pardict["xfit_min"][0] if pardict["do_corr"] else birdmodel.pardict["xfit_max"][0]
    #fitlimhex = birdmodel.pardict["xfit_min"][2] if pardict["do_corr"] else birdmodel.pardict["xfit_max"][2]

    #taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
    #chainfile = str(
    #    fmt_str
    #    % (
    #        birdmodel.pardict["fitfile"],
    #        dat_str,
    #        fitlim,
    #        fitlimhex,
    #        taylor_strs[pardict["taylor_order"]],
    #        hex_str,
    #        marg_str,
    #    )
    #)
    chainfile= 'fit_UNIT_pybird_NGC_SGC_z1_z3.hdf5'
    print(chainfile)

    # Set up the backend
    backend = emcee.backends.HDFBackend(chainfile)
    backend.reset(nwalkers, nparams)

    # Initialize the sampler
    for i, (pardict, model, data) in enumerate(zip(pardicts, birdmodel, fittingdata)):
        sampler = emcee.EnsembleSampler(nwalkers, nparams, func, args=[birdmodel, fittingdata, plt], backend=backend)

    # Run the sampler for a max of 20000 iterations. We check convergence every 100 steps and stop if
    # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
    # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
    max_iter = 40000
    index = 0
    old_tau = np.inf
    autocorr = np.empty(max_iter)
    counter = 0
    for sample in sampler.sample(begin, iterations=max_iter, progress=True):

        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        counter += 100
        print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        print("Mean Auto-Correlation time: {0:.3f}".format(autocorr[index]))

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            print("Reached Auto-Correlation time goal: %d > 100 x %.3f" % (counter, autocorr[index]))
            break
        old_tau = tau
        index += 1


def lnpost(params, birdmodel, fittingdata, plt):

    # This returns the posterior distribution which is given by the log prior plus the log likelihood
    prior = lnprior(params, birdmodel, fittingdata)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike(params, birdmodel, fittingdata, plt)
    return prior + like

def prior_transform(u, birdmodel):
    x = np.array(u)
    for i, (pardict, model, data) in enumerate(zip(pardicts, birdmodel_z1_z3_combined, fittingdata_z1_z3_combined)):
        index = fittingdata_z1_z3_combined.index(data)
        fitting_data = fittingdata_z1_z3_combined[index]
        birdmodel = birdmodel_z1_z3_combined[index]

        lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
        upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta

        #flat prior on ln10As:
        
        x[0] = (u[0] * (upper_bounds[0] - lower_bounds[0])) + lower_bounds[0]
        
        #flat prior on h

        x[1] = (u[1] * (upper_bounds[1] - lower_bounds[1])) + lower_bounds[1]

        #flat prior on Omega_cdm

        x[2] = (u[2] * (upper_bounds[2] - lower_bounds[2])) + lower_bounds[2]

        #omega_b prior from BBN
        mu, sigma = 0.02166, 0.00037
        x[3] = scipy.stats.norm.ppf(u[3], loc=mu, scale=sigma) #something something (gaussian centered on 0.02166, width of 0.00037)

        if birdmodel.pardict["do_marg"]:
        #indexed NGC z1, SGC z1, NGC z3, SGC z3
        #b1_NGC
            x[4+(index*6)] = u[4+(index*6)] * 3.0 #b1_NGC prior transform, 0-3
        #c2_NGC
            x[5+(index*6)] = (u[5+(index*6)] - 0.5) * 8.0 #c2_NGC, range from -4 to 4
        #c4_NGC
            mu, sigma = 0, 2
            x[6+(index*6)] = scipy.stats.norm.ppf(u[6+(index*6)], loc=mu, scale=sigma) #c4_NGC, Gaussian centered around 0 with width of 2
        #b1_SGC
            x[7+(index*6)] = u[7+(index*6)] * 3.0 #b1_NGC prior transform, 0-3
        #c2_SGC
            x[8+(index*6)] = (u[8+(index*6)] - 0.5) * 8.0 #c2_NGC, range from -4 to 4
        #c4_SGC
            mu, sigma = 0, 2
            x[9+(index*6)] = scipy.stats.norm.ppf(u[9+(index*6)], loc=mu, scale=sigma) #c4_NGC, Gaussian centered around 0 with width of 2

   
        else:
        #b1_NGC:
            x[4+(index*22)] = u[4+(index*22)] * 3.0 #b1_NGC, range from 0 to 3
        #c2_NGC:
            x[5+(index*22)] = (u[5+(index*22)] - 0.5) * 8.0 #c2_NGC, range from -4 to 4
        #b3_NGC:
            mu, sigma = 0, 2
            x[6+(index*22)] = scipy.stats.norm.ppf(u[6+(index*22)], loc=mu, scale=sigma) #b3_NGC, gaussian of mu = 0, sigma = 2
        #c4_NGC:
            mu, sigma = 0, 2
            x[7+(index*22)] = scipy.stats.norm.ppf(u[7+(index*22)], loc=mu, scale=sigma) #c4_NGC, gaussian of mu = 0, sigma = 2
        #cct_NGC:
            mu, sigma = 0, 2
            x[8+(index*22)] = scipy.stats.norm.ppf(u[8+(index*22)], loc=mu, scale=sigma) #cct_NGC, gaussian of mu = 0, sigma = 2
        #cr1_NGC:
            mu, sigma = 0, 4
            x[9+(index*22)] = scipy.stats.norm.ppf(u[9+(index*22)], loc=mu, scale=sigma) #cr1_NGC, gaussian of mu = 0, sigma = 4
        #cr2_NGC:
            mu, sigma = 0, 4
            x[10+(index*22)] = scipy.stats.norm.ppf(u[10+(index*22)], loc=mu, scale=sigma) #cr2_NGC, gaussian of mu 0, sigma = 4
        #ce1_NGC:
            mu, sigma = 0, 2
            x[11+(index*22)] = scipy.stats.norm.ppf(u[11+(index*22)], loc=mu, scale=sigma) #ce1_NGC, gaussian of mu = 0, sigma = 2
        #cemono_NGC:
            mu, sigma = 0, 2
            x[12+(index*22)] = scipy.stats.norm.ppf(u[12+(index*22)], loc=mu, scale=sigma)  #cemono_NGC, gaussian of mu = 0, sigma = 2
        #cequad_NGC:
            mu, sigma = 0, 2
            x[13+(index*22)] = scipy.stats.norm.ppf(u[13+(index*22)], loc=mu, scale=sigma) #cequad_NGC, gaussian of mu = 0, sigma = 2
        #bnlo_NGC:
            mu, sigma = 0, 2
            x[14+(index*22)] = scipy.stats.norm.ppf(u[14+(index*22)], loc=mu, scale=sigma) #bnlo_NGC, gaussian of mu = 0, sigma = 2
#----------------------------------------------------------------------------------------------------------------
        #b1_SGC:
            x[15+(index*22)] = u[15+(index*22)] * 3.0 #b1_SGC, range from 0 to 3
        #c2_SGC:
            x[16+(index*22)] = (u[16+(index*22)] - 0.5) * 8.0 #c2_SGC, range from -4 to 4
        #b3_SGC:
            mu, sigma = 0, 2
            x[17+(index*22)] = scipy.stats.norm.ppf(u[17+(index*22)], loc=mu, scale=sigma) #b3_SGC, gaussian of mu = 0, sigma = 2
        #c4_SGC:
            mu, sigma = 0, 2
            x[18+(index*22)] = scipy.stats.norm.ppf(u[18+(index*22)], loc=mu, scale=sigma) #c4_SGC, gaussian of mu = 0, sigma = 2
        #cct_SGC:
            mu, sigma = 0, 2
            x[19+(index*22)] = scipy.stats.norm.ppf(u[19+(index*22)], loc=mu, scale=sigma) #cct_SGC, gaussian of mu = 0, sigma = 2
        #cr1_SGC:
            mu, sigma = 0, 4
            x[20+(index*22)] = scipy.stats.norm.ppf(u[20+(index*22)], loc=mu, scale=sigma) #cr1_SGC, gaussian of mu = 0, sigma = 4
        #cr2_SGC:
            mu, sigma = 0, 4
            x[21+(index*22)] = scipy.stats.norm.ppf(u[21+(index*22)], loc=mu, scale=sigma) #cr2_SGC, gaussian of mu 0, sigma = 4
        #ce1_SGC:
            mu, sigma = 0, 2
            x[22+(index*22)] = scipy.stats.norm.ppf(u[22+(index*22)], loc=mu, scale=sigma) #ce1_SGC, gaussian of mu = 0, sigma = 2
        #cemono_SGC:
            mu, sigma = 0, 2
            x[23+(index*22)] = scipy.stats.norm.ppf(u[23+(index*22)], loc=mu, scale=sigma)  #cemono_SGC, gaussian of mu = 0, sigma = 2
        #cequad_SGC:
            mu, sigma = 0, 2
            x[24+(index*22)] = scipy.stats.norm.ppf(u[24+(index*22)], loc=mu, scale=sigma) #cequad_SGC, gaussian of mu = 0, sigma = 2
        #bnlo_SGC:
            mu, sigma = 0, 2
            x[15+(index*22)] = scipy.stats.norm.ppf(u[25+(index*22)], loc=mu, scale=sigma) #bnlo_SGC, gaussian of mu = 0, sigma = 2
    return x
        


def lnprior(params, birdmodel, data):
    for i, (pardict, model, data) in enumerate(zip(pardicts, birdmodel_z1_z3_combined, fittingdata_z1_z3_combined)):
        index = fittingdata_z1_z3_combined.index(data)
        fitting_data = fittingdata_z1_z3_combined[index]
        birdmodel = birdmodel_z1_z3_combined[index]
        #print("Printing pardict, model, and data from zip")
        #print(i, pardict, model, data)
        #print("--------------------------------------------")
        #print("Printing fitting data:")
        #print("NGC fitting data for index %lf" %index)
        #print(fitting_data.data["fit_data_NGC"])
        #print("SGC fitting data for index %lf" %index)
        #print(fitting_data.data["fit_data_SGC"])

        if birdmodel.pardict["do_marg"]:
            b1_NGC, c2_NGC, c4_NGC = params[(4+(6*index)):(7+(6*index))] #4-7 for z1, 10-13 for z3
            b1_SGC, c2_SGC, c4_SGC = params[(7+(6*index)):(10+(6*index))] #7-10 for z3, 13-16 for z3

        else:
            b1_NGC, c2_NGC, b3_NGC, c4_NGC, cct_NGC, cr1_NGC, cr2_NGC, ce1_NGC, cemono_NGC, cequad_NGC, bnlo_NGC = params[(4+(2*11*index)):(15+(2*11*index))] #4-15 for z1, 26-37 for z3
            b1_SGC, c2_SGC, b3_SGC, c4_SGC, cct_SGC, cr1_SGC, cr2_SGC, ce1_SGC, cemono_SGC, cequad_SGC, bnlo_SGC = params[(15+(2*11*index)):(26+(2*11*index))] #15-26 for z1, 37-48 for z3

        ln10As, h, omega_cdm, omega_b = params[:4]

        lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
        upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta

        # Flat priors for cosmological parameters
        if np.any(np.less([ln10As, h, omega_cdm, omega_b], lower_bounds)) or np.any(
            np.greater([ln10As, h, omega_cdm, omega_b], upper_bounds)
        ):
            return -np.inf

        # BBN (D/H) inspired prior on omega_b
        omega_b_prior = -0.5 * (omega_b - 0.02166) ** 2 / 0.00037 ** 2

        # Flat prior for b1
        if b1_NGC < 0.0 or b1_NGC > 3.0:
            return -np.inf
        if b1_SGC < 0.0 or b1_SGC > 3.0:
            return -np.inf

        # Flat prior for c2
        if c2_NGC < -4.0 or c2_NGC > 4.0:
            return -np.inf
        if c2_SGC < -4.0 or c2_SGC > 4.0:
            return -np.inf

        # Gaussian prior for c4
        c4_prior_NGC = -0.5 * 0.25 * c4_NGC ** 2
        c4_prior_SGC = -0.5 * 0.25 * c4_SGC ** 2
        if birdmodel.pardict["do_marg"]:

            return omega_b_prior + 2*(c4_prior_NGC) + 2*(c4_prior_SGC) #assumes c4 prior range is the same between models

        else:
            # Gaussian prior for b3 of width 2 centred on 0
            b3_prior_NGC = -0.5 * 0.25 * b3_NGC ** 2

            # Gaussian prior for cct of width 2 centred on 0
            cct_prior_NGC = -0.5 * 0.25 * cct_NGC ** 2

            # Gaussian prior for cr1 of width 4 centred on 0
            cr1_prior_NGC = -0.5 * 0.0625 * cr1_NGC ** 2

            # Gaussian prior for cr1 of width 4 centred on 0
            cr2_prior_NGC = -0.5 * 0.0625 * cr2_NGC ** 2

            # Gaussian prior for ce1 of width 2 centred on 0
            ce1_prior_NGC = -0.5 * 0.25 * ce1_NGC ** 2

            # Gaussian prior for cemono of width 2 centred on 0
            cemono_prior_NGC = -0.5 * 0.25 * cemono_NGC ** 2

            # Gaussian prior for cequad of width 2 centred on 0
            cequad_prior_NGC = -0.5 * 0.25 * cequad_NGC ** 2

            # Gaussian prior for bnlo of width 2 centred on 0
            bnlo_prior_NGC = -0.5 * 0.25 * bnlo_NGC ** 2
#---------------------------------------------------------------
            # Gaussian prior for b3 of width 2 centred on 0
            b3_prior_SGC = -0.5 * 0.25 * b3_SGC ** 2

            # Gaussian prior for cct of width 2 centred on 0
            cct_prior_SGC = -0.5 * 0.25 * cct_SGC ** 2

            # Gaussian prior for cr1 of width 4 centred on 0
            cr1_prior_SGC = -0.5 * 0.0625 * cr1_SGC ** 2

            # Gaussian prior for cr1 of width 4 centred on 0
            cr2_prior_SGC = -0.5 * 0.0625 * cr2_SGC ** 2

            # Gaussian prior for ce1 of width 2 centred on 0
            ce1_prior_SGC = -0.5 * 0.25 * ce1_SGC ** 2

            # Gaussian prior for cemono of width 2 centred on 0
            cemono_prior_SGC = -0.5 * 0.25 * cemono_SGC ** 2

            # Gaussian prior for cequad of width 2 centred on 0
            cequad_prior_SGC = -0.5 * 0.25 * cequad_SGC ** 2

            # Gaussian prior for bnlo of width 2 centred on 0
            bnlo_prior_SGC = -0.5 * 0.25 * bnlo_SGC ** 2

            return (
                omega_b_prior
                + c4_prior_NGC
                + b3_prior_NGC
                + cct_prior_NGC
                + cr1_prior_NGC
                + cr2_prior_NGC
                + ce1_prior_NGC
                + cemono_prior_NGC
                + cequad_prior_NGC
                + bnlo_prior_NGC

                + c4_prior_SGC
                + b3_prior_SGC
                + cct_prior_SGC
                + cr1_prior_SGC
                + cr2_prior_SGC
                + ce1_prior_SGC
                + cemono_prior_SGC
                + cequad_prior_SGC
                + bnlo_prior_SGC
            )


def lnlike(params, birdmodel, fittingdata, plt):
    chi_squared = 0 #initialization, going to add chi^2 terms for all redshift bins
    model_count = 0
    for i, (pardict, model, data) in enumerate(zip(pardicts, birdmodel_z1_z3_combined, fittingdata_z1_z3_combined)):
        index = fittingdata_z1_z3_combined.index(data)
        birdmodel = birdmodel_z1_z3_combined[index]
        fittingdata = fittingdata_z1_z3_combined[index]
    
        if birdmodel.pardict["do_marg"]:
           b2_NGC = (params[5+(6*index)] + params[6+(6*index)])/np.sqrt(2.0) #CHECK INDEXING
           b4_NGC = (params[5+(6*index)] - params[6+(6*index)])/np.sqrt(2.0)
           bs_NGC = [params[4 + (6*index)], b2_NGC, 0, b4_NGC, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

           b2_SGC = (params[8+(6*index)] + params[9+(6*index)])/np.sqrt(2.0)
           b4_SGC = (params[8+(6*index)] - params[9+(6*index)])/np.sqrt(2.0)
           bs_SGC = [params[7 + (6*index)], b2_SGC, 0, b4_SGC, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
           #print(bs_SGC)

        #For indexing reference:

        else:
            b2_NGC = (params[5+(2*11*index)] + params[7+(2*11*index)])/np.sqrt(2.0)
            b4_NGC = (params[5+(2*11*index)] - params[7+(2*11*index)])/np.sqrt(2.0)

            bs_NGC = [
                params[4+(2*11*index)],
                b2_NGC,
                params[6+(2*11*index)],
                b4_NGC,
                params[8+(2*11*index)],
                params[9+(2*11*index)],
                params[10+(2*11*index)],
                params[11+(2*11*index)] * fittingdata.data["shot_noise_NGC"],
                params[12+(2*11*index)] * fittingdata.data["shot_noise_NGC"],
                params[13+(2*11*index)] * fittingdata.data["shot_noise_NGC"],
                params[14+(2*11*index)],
            ]

            b2_SGC = (params[16+(2*11*index)] + params[18+(2*11*index)])/np.sqrt(2.0) 
            b4_SGC = (params[16+(2*11*index)] - params[18+(2*11*index)])/np.sqrt(2.0)
 
            bs_SGC = [
                params[15+(2*11*index)],
                b2_SGC,
                params[17+(2*11*index)],
                b4_SGC,
                params[19+(2*11*index)],
                params[20+(2*11*index)],
                params[21+(2*11*index)],
                params[22+(2*11*index)] * fittingdata.data["shot_noise_SGC"],
                params[23+(2*11*index)] * fittingdata.data["shot_noise_SGC"],
                params[24+(2*11*index)] * fittingdata.data["shot_noise_SGC"],
                params[25+(2*11*index)],
            ]

        # Get the bird model
        ln10As, h, omega_cdm, omega_b = params[:4]

        Plin_NGC, Ploop_NGC, Plin_SGC, Ploop_SGC = birdmodel.compute_pk_separategrids([ln10As, h, omega_cdm, omega_b])

        P_model, P_model_interp, P_model_NGC, P_model_interp_NGC, P_model_SGC, P_model_interp_SGC = birdmodel.compute_model_separategrids(bs_NGC, bs_SGC, Plin_NGC, Ploop_NGC, Plin_SGC, Ploop_SGC, fittingdata.data["x_data"])

        Pi_NGC, Pi_SGC, Pi = birdmodel.get_Pi_for_marg_separategrids(Ploop_NGC, Ploop_SGC, bs_NGC[0], bs_SGC[0], fittingdata.data["shot_noise_NGC"], fittingdata.data["shot_noise_SGC"], fittingdata.data["x_data"])

        chi_squared = chi_squared + birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)
        model_count = model_count + 1

        #To pass the correct models and Pi terms if running plots
        if model_count == 1:
            P_model_interp_z1 = P_model_interp
            Pi_z1 = Pi
        if model_count == 2:
            P_model_interp_z3 = P_model_interp
            Pi_z3 = Pi

#    if plt is not None and model_count == 2:
#        if birdmodel.pardict["do_marg"]:

#            bs_analytic_z1 = birdmodel.compute_bestfit_analytic(Pi_z1, fittingdata_z1_combined.data, P_model_interp_z1) #Should output NGC-SGC bias params in one object
#            bs_analytic_z3 = birdmodel.compute_bestfit_analytic(Pi_z3, fittingdata_z3_combined.data, P_model_interp_z3) 

#            b2_NGC_z1 = (params[5] + params[6])/np.sqrt(2.0)
#            b4_NGC_z1 = (params[5] - params[6])/np.sqrt(2.0)
#            bs_NGC_z1 = [
#                params[4],
#                b2_NGC_z1,
#                bs_analytic_z1[0],
#                b4_NGC_z1,
#                bs_analytic_z1[1],
#                bs_analytic_z1[2],
#                bs_analytic_z1[3],
#                bs_analytic_z1[4] * fittingdata_z1_combined.data["shot_noise_NGC"],
#                bs_analytic_z1[5] * fittingdata_z1_combined.data["shot_noise_NGC"],
#                bs_analytic_z1[6] * fittingdata_z1_combined.data["shot_noise_NGC"],
#                bs_analytic_z1[7],
#            ]
#
#            b2_SGC_z1 = (params[8] + params[9])/np.sqrt(2.0)
#            b4_SGC_z1 = (params[8] - params[9])/np.sqrt(2.0)
#
#            bs_SGC_z1 = [
#                params[7],
#                b2_SGC_z1,
#                bs_analytic_z1[8],
#                b4_SGC_z1,
#                bs_analytic_z1[9],
#                bs_analytic_z1[10],
#                bs_analytic_z1[11],
#                bs_analytic_z1[12] * fittingdata_z1_combined.data["shot_noise_SGC"],
#                bs_analytic_z1[13] * fittingdata_z1_combined.data["shot_noise_SGC"],
#                bs_analytic_z1[14] * fittingdata_z1_combined.data["shot_noise_SGC"],
#                bs_analytic_z1[15],
#            ]
#
#           #----------------------------------------------------
#            b2_NGC_z3 = (params[11] + params[12])/np.sqrt(2.0)
#            b4_NGC_z3 = (params[11] - params[12])/np.sqrt(2.0)
#
#            bs_NGC_z3 = [
#                params[10],
#                b2_NGC_z3,
#                bs_analytic_z3[0],
#                b4_NGC_z3,
#                bs_analytic_z3[1],
#                bs_analytic_z3[2],
#                bs_analytic_z3[3],
#                bs_analytic_z3[4] * fittingdata_z3_combined.data["shot_noise_NGC"],
#                bs_analytic_z3[5] * fittingdata_z3_combined.data["shot_noise_NGC"],
#                bs_analytic_z3[6] * fittingdata_z3_combined.data["shot_noise_NGC"],
#                bs_analytic_z3[7],
#            ]
#
#            b2_SGC_z3 = (params[14] + params[15])/np.sqrt(2.0)
#            b4_SGC_z3 = (params[14] - params[15])/np.sqrt(2.0)
#
#            bs_SGC_z3 = [
#                params[13],
#                b2_SGC_z3,
#                bs_analytic_z3[8],
#                b4_SGC_z3,
#                bs_analytic_z3[9],
#                bs_analytic_z3[10],
#                bs_analytic_z3[11],
#                bs_analytic_z3[12] * fittingdata_z3_combined.data["shot_noise_SGC"],
#                bs_analytic_z3[13] * fittingdata_z3_combined.data["shot_noise_SGC"],
#                bs_analytic_z3[14] * fittingdata_z3_combined.data["shot_noise_SGC"],
#                bs_analytic_z3[15],
#            ]
#
#            ln10As, h, omega_cdm, omega_b = params[:4]
#            Plin_NGC_z1, Ploop_NGC_z1, Plin_SGC_z1, Ploop_SGC_z1 = birdmodel_z1_combined.compute_pk_separategrids([ln10As, h, omega_cdm, omega_b])
#            Plin_NGC_z3, Ploop_NGC_z3, Plin_SGC_z3, Ploop_SGC_z3 = birdmodel_z3_combined.compute_pk_separategrids([ln10As, h, omega_cdm, omega_b])
#
#            P_model_z1, P_model_interp_z1, P_model_NGC_z1, P_model_interp_NGC_z1, P_model_SGC_z1, P_model_interp_SGC_z1 = birdmodel_z1_combined.compute_model_separategrids(bs_NGC_z1, bs_SGC_z1, Plin_NGC_z1, Ploop_NGC_z1, Plin_SGC_z1, Ploop_SGC_z1, fittingdata_z1_combined.data["x_data"])
#            P_model_z3, P_model_interp_z3, P_model_NGC_z3, P_model_interp_NGC_z3, P_model_SGC_z3, P_model_interp_SGC_z3 = birdmodel_z3_combined.compute_model_separategrids(bs_NGC_z3, bs_SGC_z3, Plin_NGC_z3, Ploop_NGC_z3, Plin_SGC_z3, Ploop_SGC_z3, fittingdata_z3_combined.data["x_data"])
#           
#        update_plot_NGC_SGC_z1_z3(pardict_z1_NGC, pardict_z3_NGC, fittingdata_z1_combined.data["x_data"], fittingdata_z3_combined.data["x_data"], P_model_interp_NGC_z1, P_model_interp_SGC_z1, P_model_interp_NGC_z3, P_model_interp_SGC_z3, colours, "output_NGC_SGC_z1_z3.png", plt, keep=False)

    return -0.5 * chi_squared


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    NGC_z1_configfile = "../config/tbird_NGC_z1_s10fixed_singlefit.txt"
    SGC_z1_configfile = "../config/tbird_SGC_z1_s10fixed_singlefit.txt"
    NGC_z3_configfile = "../config/tbird_NGC_z3_s10fixed_singlefit.txt"
    SGC_z3_configfile = "../config/tbird_SGC_z3_s10fixed_singlefit.txt"
    plot_flag = int(sys.argv[1])
    pardict_z1_NGC = ConfigObj(NGC_z1_configfile)
    pardict_z1_SGC = ConfigObj(SGC_z1_configfile)
    pardict_z3_NGC = ConfigObj(NGC_z3_configfile)
    pardict_z3_SGC = ConfigObj(SGC_z3_configfile)
    ndatasets = 4 #for emcee/dynesty
    #print(pardict_z1_NGC)
    #print(pardict_z1_SGC)
    #print(pardict_z3_NGC)
    #print(pardict_z3_SGC)  

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict_z1_NGC = format_pardict(pardict_z1_NGC)
    #print("pardict_z1_NGC = ")
    #print(pardict_z1_NGC)
    pardict_z1_SGC = format_pardict(pardict_z1_SGC)
    pardict_z3_NGC = format_pardict(pardict_z3_NGC)
    pardict_z3_SGC = format_pardict(pardict_z3_SGC)
    pardicts_z1 = [pardict_z1_NGC, pardict_z1_SGC]
    pardicts_z3 = [pardict_z3_NGC, pardict_z3_SGC]
    pardicts = [pardict_z1_NGC, pardict_z3_NGC] #Since pardict z1 NGC combines NGC/SGC data in the ini file, should be safe to just pass through pair of redshifts
    NGC_z1_shot_noise = float(pardict_z1_NGC["shot_noise"])
    SGC_z1_shot_noise = float(pardict_z1_SGC["shot_noise"])
    NGC_z3_shot_noise = float(pardict_z3_NGC["shot_noise"])
    SGC_z3_shot_noise = float(pardict_z3_SGC["shot_noise"])

    # Set up the data
    fittingdata_z1_combined = FittingData_NGC_SGC(pardict_z1_NGC, pardict_z1_SGC, NGC_z1_shot_noise, SGC_z1_shot_noise)
    fittingdata_z3_combined = FittingData_NGC_SGC(pardict_z3_NGC, pardict_z3_SGC, NGC_z3_shot_noise, SGC_z3_shot_noise)
    fittingdata_z1_z3_combined = [fittingdata_z1_combined, fittingdata_z3_combined]

    # Set up the BirdModel
    #Only need to pass single pardict when creating the birdmodel
    birdmodel_z1_combined = BirdModel(pardict_z1_NGC, template=False)
    birdmodel_z3_combined = BirdModel(pardict_z3_NGC, template=False)
    birdmodel_z1_z3_combined = [birdmodel_z1_combined, birdmodel_z3_combined]

    # Plotting (for checking/debugging, should turn off for production runs)
    #plt = None
    if plot_flag:
        colours = ["b", "g", "r", "c", "m", "y", "k", "mediumorchid"]
        plt = create_plot_NGC_SGC_z1_z3(pardict_z1_NGC, pardict_z1_SGC, pardict_z3_NGC, pardict_z3_SGC, colours, fittingdata_z1_combined, fittingdata_z3_combined) #NOTE: Pardict NGC must contain both gridnames for the NGC and SGC sections

    if birdmodel_z1_combined.pardict["do_marg"]:
        start = np.concatenate([birdmodel_z1_combined.valueref[:4], [1.3, 0.5, 0.5], [1.3, 0.5, 0.5]])
    else:
        start = np.concatenate([birdmodel_z1_combined.valueref[:4], [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

    if birdmodel_z3_combined.pardict["do_marg"]:
        start = np.concatenate([start, [1.3, 0.5, 0.5], [1.3, 0.5, 0.5]])
    else:
        start = np.concatenate([start, [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

    #Our data vector follows the order [[4 cosmo], [z1_NGC], [z1_SGC], [z3_NGC], [z3_SGC]]

    # Does an optimization
    #result = do_optimization(lambda *args: -lnpost(*args), start, birdmodel_z1_z3_combined, fittingdata_z1_z3_combined, plt)

    # Does an MCMC
    #do_emcee(lnpost, start, birdmodel_z1_z3_combined, fittingdata_z1_z3_combined, plt)

    # Calculates evidences using Dynesty
    if birdmodel_z1_combined.pardict["do_marg"]:
        ndim = 4+(3*ndatasets)
    else:
        ndim = 4+(11*ndatasets)
    
    #log_like = lnlike(start, birdmodel, fittingdata, plt)
    dsampler = DynamicNestedSampler(lnlike, prior_transform, ndim, logl_args=[birdmodel_z1_z3_combined, fittingdata_z1_z3_combined, plt], ptform_args=[birdmodel_z1_z3_combined]) #Loglikelihood function, prior transform, and the number of dimensions taken by log likelihood
    dsampler.run_nested()
    dres = dsampler.results #TO DO: MANUALLY SAVE CHAINS AS HDF5 OR .TXT
    fig, axes = dyplot.traceplot(dres, trace_cmap='viridis', connect=True)
    
    plt.savefig("saved_dynesty_NGC_SGC_z1_z3.pdf")

    
