import numpy as np
import sys
import dynesty
import scipy
from configobj import ConfigObj
from dynesty import plotting as dyplot
from dynesty import DynamicNestedSampler

sys.path.append("../")
from fitting_codes.fitting_utils import (
    FittingData,
    BirdModel,
    create_plot,
    update_plot,
    format_pardict,
    do_optimization,
)

import matplotlib
import matplotlib.pyplot as plt

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
    prior = prior_transform(params, birdmodel)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike(params, birdmodel, fittingdata, plt)
    return prior + like


#We must map our priors from a unit hypercube to our actual priors.
#Pass prior u, which is an array u = [u1, u2, u3, ...] for each of 
#our bias params. These bias params can then be passed into our standard lnprior
#function. 

#TO DO: Need to update this if I want to run sim fits with shared cosmo params, independent
#bias params
def prior_transform(u, birdmodel):
    x = np.array(u) #this copies the array of u
    lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
    upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta

    #b1, c2, c4 = params[-3:], + cosmological priors (flat over the upper and lower bounds) [ln10As, h, omega_cdm, omega_b, Omega_k]

    #flat prior on ln10As:
        
    x[0] = (u[0] * (upper_bounds[0] - lower_bounds[0])) + lower_bounds[0]
        
    #flat prior on h

    x[1] = (u[1] * (upper_bounds[1] - lower_bounds[1])) + lower_bounds[1]

    #flat prior on Omega_cdm

    x[2] = (u[2] * (upper_bounds[2] - lower_bounds[2])) + lower_bounds[2]

    #omega_b prior from BBN
    mu, sigma = 0.02235, 0.00049
    x[3] = scipy.stats.norm.ppf(u[3], loc=mu, scale=sigma) # (gaussian centered on 0.02235, width of 0.00049)

    #flat prior on Omega_k
    x[4] = (u[4] * (upper_bounds[4] - lower_bounds[4])) + lower_bounds[4]

    if birdmodel.pardict["do_marg"]:
    
    #b1:
        x[5] = u[5] * 3.0 #b1, range from 0 to 3
    
    #c2: 
        x[6] = (u[6] - 0.5) * 8.0 #c2, range from -4 to 4

    #c4:
        mu, sigma = 0, 2
        x[7] = scipy.stats.norm.ppf(u[7], loc=mu, scale=sigma)#Gaussian centered around 0 with width of 2

    else:
        #b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad, bnlo = params[-11:]
    #b1:
        x[4] = u[4] * 3.0 #b1, range from 0 to 3
    #c2:
        x[5] = (u[5] - 0.5) * 8.0 #c2, range from -4 to 4
    #b3:
        mu, sigma = 0, 2
        x[6] = scipy.stats.norm.ppf(u[6], loc=mu, scale=sigma) #b3, gaussian of mu = 0, sigma = 2
    #c4:
        mu, sigma = 0, 2
        x[7] = scipy.stats.norm.ppf(u[7], loc=mu, scale=sigma) #c4, gaussian of mu = 0, sigma = 2
    #cct:
        mu, sigma = 0, 2
        x[8] = scipy.stats.norm.ppf(u[8], loc=mu, scale=sigma) #cct, gaussian of mu = 0, sigma = 2
    #cr1:
        mu, sigma = 0, 4
        x[9] = scipy.stats.norm.ppf(u[9], loc=mu, scale=sigma) #cr1, gaussian of mu = 0, sigma = 4
    #cr2:
        mu, sigma = 0, 4
        x[10] = scipy.stats.norm.ppf(u[10], loc=mu, scale=sigma) #cr2, gaussian of mu 0, sigma = 4
    #ce1:
        mu, sigma = 0, 2
        x[11] = scipy.stats.norm.ppf(u[11], loc=mu, scale=sigma) #ce1, gaussian of mu = 0, sigma = 2
    #cemono:
        mu, sigma = 0, 2
        x[12] = scipy.stats.norm.ppf(u[12], loc=mu, scale=sigma)  #cemono, gaussian of mu = 0, sigma = 2
    #cequad:
        mu, sigma = 0, 2
        x[13] = scipy.stats.norm.ppf(u[13], loc=mu, scale=sigma) #cemono, gaussian of mu = 0, sigma = 2
    #bnlo:
        mu, sigma = 0, 2
        x[14] = scipy.stats.norm.ppf(u[14], loc=mu, scale=sigma) #bnlo, gaussian of mu = 0, sigma = 2
    #print("returned prior_transform = ")
    #print(x)
    return x
'''
def lnprior(params, birdmodel, fittingdata, plt):

    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors
    if birdmodel.pardict["do_marg"]:
        b1, c2, c4 = params[-3:]
    else:
        b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad, bnlo = params[-11:]

    ln10As, h, omega_cdm, omega_b = params[:4]

    lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
    upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta

    # Flat priors for cosmological parameters
    if np.any(np.less([ln10As, h, omega_cdm, omega_b], lower_bounds)) or np.any(
        np.greater([ln10As, h, omega_cdm, omega_b], upper_bounds)
    ):
        return -np.inf

    # BBN (D/H) inspired prior on omega_b
    #omega_b_prior = -0.5 * (omega_b - birdmodel.valueref[3]) ** 2 / 0.00037 ** 2
    omega_b_prior = -0.5 * (omega_b - 0.02166) ** 2 / 0.00037 ** 2

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

'''
def lnlike(params, birdmodel, fittingdata, plt): #just testing whether dynesty needs only theta of variables passed through lnlike and ptransform

    #birdmodel, fittingdata, plt = birdmodel, fittingdata, plt

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
    #print("params passed to lnlike = ")
    #print(params)
    #print("vector of bias params defined!")

    # Get the bird model
    ln10As, h, omega_cdm, omega_b, omega_k = params[:5]
    #print("cosmo variables called from params")

    Plin, Ploop = birdmodel.compute_pk([ln10As, h, omega_cdm, omega_b, omega_k])
    #print("Plin, Ploop calculated")
    P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
    Pi = birdmodel.get_Pi_for_marg(Ploop, bs[0], fittingdata.data["shot_noise"], fittingdata.data["x_data"])

    chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)
    #print("chi_squared calculated")


#    if plt is not None:
#        chi_squared_print = chi_squared
#        if birdmodel.pardict["do_marg"]:
#            bs_analytic = birdmodel.compute_bestfit_analytic(Pi, fittingdata.data, P_model_interp)
#            pardict["do_marg"] = 0
#            b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
#            b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
#            bs = [
#                params[-3],
#                b2,
#                bs_analytic[0],
#                b4,
#                bs_analytic[1],
#                bs_analytic[2],
#                bs_analytic[3],
#                bs_analytic[4] * fittingdata.data["shot_noise"],
#                bs_analytic[5] * fittingdata.data["shot_noise"],
#                bs_analytic[6] * fittingdata.data["shot_noise"],
#                bs_analytic[7],
#            ]
#            P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
#            chi_squared_print = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)
#            pardict["do_marg"] = 1
#        update_plot(pardict, fittingdata.data["x_data"], P_model_interp, plt)
#        if np.random.rand() < 0.1:
#            print(params, chi_squared_print)

    #print("log likelihood calculated to be : %lf" %(-0.5*chi_squared))
    return -0.5 * chi_squared


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    #configfile = sys.argv[1]
    #configfile = "../config/tbird_NGC_z1_s10fixed_singlefit_singlecov_curved_nbodykit.txt"
    #configfile = "../config/tbird_SGC_z1_s10fixed_singlefit_curved_nbodykit.txt"
    #configfile = "../config/tbird_NGC_z3_s10fixed_singlefit_singlecov_curved_nbodykit.txt"
    configfile = "../config/tbird_SGC_z3_s10fixed_singlefit_curved_nbodykit.txt"
    plot_flag = int(sys.argv[1])
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)

    # Set up the data
    fittingdata = FittingData(pardict, shot_noise=float(pardict["shot_noise"]))

    # Set up the BirdModel
    birdmodel = BirdModel(pardict, template=False)

    # Plotting (for checking/debugging, should turn off for production runs)
    #plt = None
    #if plot_flag:
    #    plt = create_plot(pardict, fittingdata)

    if birdmodel.pardict["do_marg"]:
        start = np.concatenate([birdmodel.valueref[:5], [1.3, 0.5, 0.5]]) #5 to include curvature as free parameter
    else:
        start = np.concatenate([birdmodel.valueref[:5], [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

    # Does an optimization
    #result = do_optimization(lambda *args: -lnpost(*args), start, birdmodel, fittingdata, plt)

    # Does an MCMC
    #do_emcee(lnpost, start, birdmodel, fittingdata, plt)

    # Calculates evidences using Dynesty
    if birdmodel.pardict["do_marg"]:
        ndim = 8
    else:
        ndim = 15
    
    #log_like = lnlike(start, birdmodel, fittingdata, plt)
    dsampler = DynamicNestedSampler(lnlike, prior_transform, ndim, logl_args=[birdmodel, fittingdata, plt], ptform_args=[birdmodel]) #Loglikelihood function, prior transform, and the number of dimensions taken by log likelihood
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
    np.savetxt("dynesty_SGC_z3_nbodykit_om_2p235_err_49_curved.dat", output)

    #plt.cla()
    #plt.clf()
    #plt.close()
    #fig, ax = plt.subplots(figsize=(10,8))
    #plt.axhline(y=0.1)
    #plt.show()
    fig, axes = dyplot.traceplot(dres, trace_cmap='viridis', connect=True)
    
    plt.savefig("dynesty_SGC_z3_nbodykit_om_2p235_err_49_curved.pdf")


