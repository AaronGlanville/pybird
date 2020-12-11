import numpy as np
import sys
from configobj import ConfigObj

sys.path.append("../")
from fitting_codes.fitting_utils_NGCSGC_Combined import (
    FittingData_NGC_SGC,
    BirdModel,
    create_plot_combined,
    update_plot_combined,
    update_plot_individual,
    format_pardict,
    do_optimization,
)

def do_emcee(func, start, birdmodel, fittingdata, plt):

    import emcee

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    nparams = len(start)
    nwalkers = nparams*2
    print("nparams")
    print(nparams)
    print("nwalkers")
    print(nwalkers)
    print("start")

    begin = [[(0.01 * (np.random.rand() - 0.5) + 1.0) * start[j] for j in range(len(start))] for i in range(nwalkers)]

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
    chainfile= 'fit_UNIT_pybird_NGC_SGC.hdf5'
    print(chainfile)

    # Set up the backend
    backend = emcee.backends.HDFBackend(chainfile)
    backend.reset(nwalkers, nparams)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, nparams, func, args=[birdmodel, fittingdata, plt], backend=backend)

    # Run the sampler for a max of 20000 iterations. We check convergence every 100 steps and stop if
    # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
    # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
    max_iter = 20000
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
    prior = lnprior(params, birdmodel)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike(params, birdmodel, fittingdata, plt)
    return prior + like


def lnprior(params, birdmodel):

    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors
    if birdmodel.pardict["do_marg"]:
        b1_NGC, c2_NGC, c4_NGC = params[4:7] #in order [cosmo], [b_NGC], [b_SGC]
        b1_SGC, c2_SGC, c4_SGC = params[7:10]
    else:
        b1_NGC, c2_NGC, b3_NGC, c4_NGC, cct_NGC, cr1_NGC, cr2_NGC, ce1_NGC, cemono_NGC, cequad_NGC, bnlo_NGC = params[4:15]
        b1_SGC, c2_SGC, b3_SGC, c4_SGC, cct_SGC, cr1_SGC, cr2_SGC, ce1_SGC, cemono_SGC, cequad_SGC, bnlo_SGC = params[15:26]

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

        return omega_b_prior + c4_prior_NGC + c4_prior_SGC

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
    #print(chi_squared)

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

    return -0.5 * chi_squared


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    NGC_configfile = "../config/tbird_NGC_z1_s10fixed_singlefit.txt"
    SGC_configfile = "../config/tbird_SGC_z1_s10fixed_singlefit.txt"
    #NGC_configfile = "../config/tbird_NGC_z3_s10fixed_singlefit.txt"
    #SGC_configfile = "../config/tbird_SGC_z3_s10fixed_singlefit.txt"
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
    plt = None
    if plot_flag:
        colours = ["b", "g", "r", "c", "n"]
        plt = create_plot_combined(pardict_NGC, pardict_SGC, colours, fittingdata_combined) #NOTE: Pardict NGC must contain both gridnames for the NGC and SGC sections

    if birdmodel_combined.pardict["do_marg"]:
        start = np.concatenate([birdmodel_combined.valueref[:4], [1.3, 0.5, 0.5], [1.3, 0.5, 0.5]])
    else:
        start = np.concatenate([birdmodel_combined.valueref[:4], [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

    # Does an optimization
    result = do_optimization(lambda *args: -lnpost(*args), start, birdmodel_combined, fittingdata_combined, plt)

    # Does an MCMC
    #do_emcee(lnpost, start, birdmodel_combined, fittingdata_combined, plt)
