import numpy as np
import sys
from configobj import ConfigObj

sys.path.append("../")
from fitting_codes.fitting_utils import (
    FittingData,
    BirdModel,
    create_plot,
    update_plot,
    format_pardict,
    do_optimization,
)

def do_emcee(func, start, birdmodel, fittingdata, plt):

    import emcee

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    nparams = len(start)
    nwalkers = nparams * 8

    begin = [[(0.01 * (np.random.rand() - 0.5) + 1.0) * start[j] for j in range(len(start))] for i in range(nwalkers)]

    marg_str = "marg" if pardict["do_marg"] else "all"
    hex_str = "hex" if pardict["do_hex"] else "nohex"
    dat_str = "xi" if pardict["do_corr"] else "pk"
    fmt_str = "%s_%s_%2dhex%2d_%s_%s_%s.hdf5" if pardict["do_corr"] else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s.hdf5"
    fitlim = birdmodel.pardict["xfit_min"][0] if pardict["do_corr"] else birdmodel.pardict["xfit_max"][0]
    fitlimhex = birdmodel.pardict["xfit_min"][2] if pardict["do_corr"] else birdmodel.pardict["xfit_max"][2]

    taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
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
    

    chainfile = "SGC_z1_marg_MCMC_fit_curved_nbodykit_wide_grid_fixed_ob"
    #chainfile = "test"
    print(chainfile)

    # Set up the backend
    backend = emcee.backends.HDFBackend(chainfile)
    backend.reset(nwalkers, nparams)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, nparams, func, args=[birdmodel, fittingdata, plt], backend=backend)

    # Run the sampler for a max of 20000 iterations. We check convergence every 100 steps and stop if
    # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
    # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
    max_iter = 50000
    index = 0
    old_tau = np.inf
    autocorr = np.empty(max_iter)
    counter = 0
    #for sample in sampler.sample(begin, skip_initial_state_check = True, iterations=max_iter, progress=True):
    for sample in sampler.sample(begin, skip_initial_state_check = True, iterations=max_iter, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        counter += 100
        #print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        #print("Mean Auto-Correlation time: {0:.3f}".format(autocorr[index]))

        print("converged if")
        print(tau) 
        print("is less than %lf" %(np.float(sampler.iteration)/100))
        print("and if difference")
        print(np.abs(old_tau-tau)/(tau))
        print("is less than 0.01")

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            print("Reached Auto-Correlation time goal: %d > 100 x %.3f" % (counter, autocorr[index]))
            break
        old_tau = tau

    #if sampler.iteration == (max_iter - 1):
    #    print("End of iteration reached, printing:")
    #    break

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
        b1, c2, c4 = params[-3:]
    else:
        b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad, bnlo = params[-11:]

    ln10As, h, omega_cdm, omega_b, Omega_k = params[:5]

    lower_bounds = birdmodel.valueref - np.abs(birdmodel.pardict["order"] * birdmodel.delta)
    upper_bounds = birdmodel.valueref + np.abs(birdmodel.pardict["order"] * birdmodel.delta)

    # Flat priors for cosmological parameters
    if np.any(np.less([ln10As, h, omega_cdm, omega_b, Omega_k], lower_bounds)) or np.any(
        np.greater([ln10As, h, omega_cdm, omega_b, Omega_k], upper_bounds)
    ):
        #print("Upper bounds = ")
        #print(upper_bounds)
        #print("Lower bounds = ")
        #print(lower_bounds)
        #print("prior is not happy")
        #print([ln10As, h, omega_cdm, omega_b, Omega_k])

        return -np.inf

    # BBN (D/H) inspired prior on omega_b
    #omega_b_prior = -0.5 * (omega_b - birdmodel.valueref[3]) ** 2 / 0.00037 ** 2
    omega_b_prior = -0.5 * (omega_b - 0.02166) ** 2 / 0.00026 ** 2 #updated prior

    # Flat prior for b1
    if b1 < 0.0 or b1 > 3.0:
        return -np.inf

    # Flat prior for c2
    if c2 < -4.0 or c2 > 4.0:
        return -np.inf

    # Gaussian prior for c4
    c4_prior = -0.5 * 0.25 * c4 ** 2

    if birdmodel.pardict["do_marg"]:

        return omega_b_prior + c4_prior

    else:
        # Gaussian prior for b3 of width 2 centred on 0
        b3_prior = -0.5 * 0.25 * b3 ** 2

        # Gaussian prior for cct of width 2 centred on 0
        cct_prior = -0.5 * 0.25 * cct ** 2

        # Gaussian prior for cr1 of width 4 centred on 0
        cr1_prior = -0.5 * 0.0625 * cr1 ** 2

        # Gaussian prior for cr1 of width 4 centred on 0
        cr2_prior = -0.5 * 0.0625 * cr2 ** 2

        # Gaussian prior for ce1 of width 2 centred on 0
        ce1_prior = -0.5 * 0.25 * ce1 ** 2

        # Gaussian prior for cemono of width 2 centred on 0
        cemono_prior = -0.5 * 0.25 * cemono ** 2

        # Gaussian prior for cequad of width 2 centred on 0
        cequad_prior = -0.5 * 0.25 * cequad ** 2

        # Gaussian prior for bnlo of width 2 centred on 0
        bnlo_prior = -0.5 * 0.25 * bnlo ** 2

        return (
            omega_b_prior
            + c4_prior
            + b3_prior
            + cct_prior
            + cr1_prior
            + cr2_prior
            + ce1_prior
            + cemono_prior
            + cequad_prior
            + bnlo_prior
        )


def lnlike(params, birdmodel, fittingdata, plt):

    if birdmodel.pardict["do_marg"]:
        b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
        b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
        bs = [params[-3], b2, 0.0, b4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        b2 = (params[-10] + params[-8]) / np.sqrt(2.0)
        b4 = (params[-10] - params[-8]) / np.sqrt(2.0)
        bs = [
            params[-11],
            b2,
            params[-9],
            b4,
            params[-7],
            params[-6],
            params[-5],
            params[-4] * fittingdata.data["shot_noise"],
            params[-3] * fittingdata.data["shot_noise"],
            params[-2] * fittingdata.data["shot_noise"],
            params[-1],
        ]

    # Get the bird model
    ln10As, h, omega_cdm, omega_b, Omega_k = params[:5]

    Plin, Ploop = birdmodel.compute_pk([ln10As, h, omega_cdm, omega_b, Omega_k])
    P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
    Pi = birdmodel.get_Pi_for_marg(Ploop, bs[0], fittingdata.data["shot_noise"], fittingdata.data["x_data"])

    chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)

    if plt is not None:
        chi_squared_print = chi_squared
        if birdmodel.pardict["do_marg"]:
            bs_analytic = birdmodel.compute_bestfit_analytic(Pi, fittingdata.data, P_model_interp)
            pardict["do_marg"] = 0
            b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
            b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
            bs = [
                params[-3],
                b2,
                bs_analytic[0],
                b4,
                bs_analytic[1],
                bs_analytic[2],
                bs_analytic[3],
                bs_analytic[4] * fittingdata.data["shot_noise"],
                bs_analytic[5] * fittingdata.data["shot_noise"],
                bs_analytic[6] * fittingdata.data["shot_noise"],
                bs_analytic[7],
            ]
            P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
            chi_squared_print = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)
            pardict["do_marg"] = 1
        update_plot(pardict, fittingdata.data["x_data"], P_model_interp, plt)
        #if np.random.rand() < 0.1:
        #    print(params, chi_squared_print)

    return -0.5 * chi_squared


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    #configfile = "../config/tbird_NGC_z3_s10fixed_singlefit_singlecov_curved.txt"
    #configfile = "../config/tbird_SGC_z3_s10fixed_singlefit_curved_omk0p2_center.txt" #SGC fits using a grid centered on Omega_k = +0.02, instead of the standard -0.02.
    configfile = "../config/tbird_SGC_z1_s10fixed_singlefit_curved_nbodykit.txt" #SGC z1, using nbodykit Pk and cov_matrix, wider grid
    #configfile = "../config/tbird_SGC_z3_s10fixed_singlefit_curved_omk0p2_center.txt"
    plot_flag = int(sys.argv[1])
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)
    print("pardict = ")
    print(pardict)

    # Set up the data
    fittingdata = FittingData(pardict, shot_noise=float(pardict["shot_noise"]))

    # Set up the BirdModel
    birdmodel = BirdModel(pardict, template=False)
    print("Birdmodel is ")
    print(birdmodel)

    # Plotting (for checking/debugging, should turn off for production runs)
    plt = None
    if plot_flag:
        plt = create_plot(pardict, fittingdata)

    if birdmodel.pardict["do_marg"]:
        start = np.concatenate([birdmodel.valueref[:5], [1.3, 0.5, 0.5]])
    else:
        start = np.concatenate([birdmodel.valueref[:5], [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    print(start)

    # Does an optimization
    #result = do_optimization(lambda *args: -lnpost(*args), start, birdmodel, fittingdata, plt)
#[2.89498558, 0.66985936, 0.12254694, 0.02165628, 0.06544618, 2.15989887, 0.43412161, 0.09590715] -NGC z3 BEST FIT: SHOULD CHECK THIS- Maybe repeat with omega_k set to positive value
# 2.24184294,  0.68552287,  0.1077006 ,  0.02165721, -0.14, 2.59533951, -1.99523625,  0.03326179] -SGC z3 BEST FIT: Going to repeat NGC z3 with Omega_k base set to positive to check indexing
# [3.05986046, 0.66211256, 0.12489157, 0.02166699, 0.13999985, 2.10330561, 0.4033078 , 0.17462283] -NGC z3 BEST FIT, set omega_k center to +0.02- still pushing to edge of grid, no clue why? Seems like Omega_k degenerate with As, maybe need to place a physically motivated prior?
    # Does an MCMC
    do_emcee(lnpost, start, birdmodel, fittingdata, plt)
