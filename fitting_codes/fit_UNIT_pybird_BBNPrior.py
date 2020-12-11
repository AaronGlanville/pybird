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
    chainfile = str(
        fmt_str
        % (
            birdmodel.pardict["fitfile"],
            dat_str,
            fitlim,
            fitlimhex,
            taylor_strs[pardict["taylor_order"]],
            hex_str,
            marg_str,
        )
    )
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

        #print(sample)
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
        b1, c2, c4 = params[-3:]
    else:
        b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad, bnlo = params[-11:]

    #if birdmodel.pardict["free_Omega_b"]:
    ln10As, h, omega_cdm, omega_b = params[:4]
    
    
    #else:
    # BBN (D/H) inspired prior on omega_b
    #    ln10As, h, omega_cdm = params[:3]
    #    omega_b_prior = -0.5 * (omega_b - birdmodel.valueref[3]) ** 2 / 0.00037 ** 2
    #    omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm

    lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
    upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta

    #print("Lower Bounds")
    #print(lower_bounds)
    #print("Upper Bounds")
    #print(upper_bounds)

    # Flat priors for cosmological parameters
    if np.any(np.less([ln10As, h, omega_cdm, omega_b], lower_bounds)) or np.any(
        np.greater([ln10As, h, omega_cdm, omega_b], upper_bounds)
    ):
        return -np.inf

    #Additional prior on Omega_B
    #omega_b_prior = -0.5 * ((omega_b - 0.02166)**2)/(0.00019**2) #BASED ON COOKE 2018, ERRORS COMBINED IN QUADRATURE
    omega_b_prior = -0.5 * ((omega_b - 0.02166)**2)/(0.00027**2) #BASED ON COOKE 2018, ERRORS COMBINED LINEARLY
    #omega_b_prior = -0.5 * ((omega_b - 0.02235)**2)/(0.0005**2) #Prior used in D'Amico limits of wCDM cosmology, page 5
#DEFINE NUISANCE PARAMETERS IN INI FILE

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

#DEFINE PRIOR LIST IN EACH INI
        
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

    #Get the bird model
    #if birdmodel.pardict["BBN_Omega_b"]:
    ln10As, h, omega_cdm, omega_b = params[:4]
    
    #else: #Assumed constant Omega_b/Omega_cdm ratio
    #    ln10As, h, omega_cdm = params[:3]
    #    omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm

    Plin, Ploop = birdmodel.compute_pk([ln10As, h, omega_cdm, omega_b])
    P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
    Pi = birdmodel.get_Pi_for_marg(Ploop, bs[0], fittingdata.data["shot_noise"], fittingdata.data["x_data"])

    chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)

    if plt is not None:
        update_plot(pardict, fittingdata.data["x_data"], P_model_interp, plt)
        if np.random.rand() < 0.1:
            print(params, chi_squared)

    #print(chi_squared)
    return -0.5 * chi_squared


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    plot_flag = int(sys.argv[2])
    pardict = ConfigObj(configfile)

#LIST EVERY INDIVIDUAL PARDICT

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)

#FORMAT EVERY INDIVIDUAL INI

    # Set up the data
    fittingdata = FittingData(pardict, shot_noise=float(pardict["shot_noise"]))

#CREATE LIST OF FITTING DATA

    # Set up the BirdModel
    birdmodel = BirdModel(pardict, template=False)

#CREATE LIST OF BIRDMODELS FOR EACH PARDICT

    # Plotting (for checking/debugging, should turn off for production runs)
    plt = None
    if plot_flag:
        plt = create_plot(pardict, fittingdata)

#DO THIS FOR EVERY INDIVIDUAL BIRDMODEL
#USE FOR LOOP, DEFINE BIASES FOR EACH INDIVIDUAL (BOTH COSMO AND SKY PATCH)

    if birdmodel.pardict["do_marg"]:
        #start = np.concatenate([birdmodel.valueref[:4], [1.3, 0.5, 0.5, 0.5]])
        start = np.concatenate([birdmodel.valueref[:4], [1.3, 0.5, 0.5]])
    else:
        start = np.concatenate([birdmodel.valueref[:4], [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

    # Does an optimization
    #result = do_optimization(lambda *args: -lnpost(*args), start, birdmodel, fittingdata, plt)

#DO EMCEE FOR LIST OF BIRDMODELS, FITTINGDATA [BIRDMODEL1, BIRDMODEL2 ETC.]

    # Does an MCMC
    do_emcee(lnpost, start, birdmodel, fittingdata, plt)