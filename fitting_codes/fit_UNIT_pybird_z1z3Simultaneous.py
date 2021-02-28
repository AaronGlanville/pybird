import numpy as np
import sys
from configobj import ConfigObj
import os

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
    for i, (pardict, model, data) in enumerate(zip(pardicts, birdmodel, fittingdata)):
        print(i, pardict, start, model, data)
        print("model:")
        print(model)
        nparams = len(start)
        nwalkers = nparams*2
        print("nparams")
        print(nparams)
        print("nwalkers")
        print(nwalkers)
        print("start")
        print(np.shape(start))
        begin = begin = [[(0.01 * (np.random.rand() - 0.5) + 1.0) * start[n] for n in range(len(start))] for i in range(nwalkers)]
        print(begin)
    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    #nparams = len(start_BOSS_NGC_z1)
    #nparams = len(start_BOSS_NGC_z1) #(4 cosmo (shared), 2*3 bias (marginalised))
    #nwalkers = nparams* n_datasets * 8
    #print("nparams:")
    #print(nparams)
    #print("nwalkers:")
    #print(nwalkers)
    #print("start")
    #print(np.shape(start))
    #begin = [[[(0.01 * (np.random.rand() - 0.5) + 1.0) * start[n][m] for n in range(7)] for m in range(2)] for i in range(nwalkers)]
    #print(begin)
    
    chainfile = 'TestBOSSNGC_z1_z3_CombinedFit_WinFunc_s1fixed_40000Steps.hdf5'
    print(chainfile)

    # Set up the backend
    backend = emcee.backends.HDFBackend(chainfile)
    backend.reset(nwalkers, nparams)
    # Initialize the sampler
    for i, (pardict, model, data) in enumerate(zip(pardicts, birdmodel, fittingdata)):
        #print(i, pardict, start, model, data)
        #for i in range(n_datasets):
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
    #print("Passed Birdmodel")
    #print(birdmodel)
    #print("Test to find corresponding pardict for dataset")
    #print(pardicts[index])
    # This returns the posterior distribution which is given by the log prior plus the log likelihood
    prior = lnprior(params, birdmodel, fittingdata)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike(params, birdmodel, fittingdata, plt)
    return prior + like
    


def lnprior(params, birdmodel, data):
    for i, (pardict, model, data) in enumerate(zip(pardicts, birdmodels, fittingdatas)):
        index = fittingdatas.index(data)
        birdmodel = birdmodels[index]
        if birdmodel.pardict["do_marg"]:
            b1, c2, c4 = params[4+(3*index):(4+(3*(index + 1)))]
        else:
            b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad, bnlo = params[-11:]
            
    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors
    #if birdmodel.pardict["do_marg"]:
    #    b1, c2, c4 = params[-3:]
    #else:
    #    b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad, bnlo = params[-11:]

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
        omega_b_prior = -0.5 * ((omega_b - 0.02166)**2)/(0.00019**2) #BASED ON COOKE 2018, ERRORS COMBINED IN QUADRATURE

#DEFINE NUISANCE PARAMETERS IN INI FILE
        b1_min = birdmodel.pardict["b1_min"]
        #print("b1_min = %s" %float(b1_min))
        b1_max = birdmodel.pardict["b1_max"]
        c2_min = birdmodel.pardict["c2_min"]
        c2_max = birdmodel.pardict["c2_max"]
        c4_width = birdmodel.pardict["c4_width"]
    # Flat prior for b1
        if b1 < float(b1_min) or b1 > float(b1_max):
            return -np.inf

    # Flat prior for c2
        if c2 < float(c2_min) or c2 > float(c2_max):
            return -np.inf

    # Gaussian prior for c4
        c4_prior = -0.5 * float(c4_width)* c4 ** 2
        
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
    chi_squared = 0 #initialisation
    for i, (pardict, model, data) in enumerate(zip(pardicts, birdmodels, fittingdatas)):
        index = fittingdatas.index(data)
        #print("INDEX")
        #print(index)
        #print(pardicts[index])
        birdmodel = birdmodels[index]
        #print("Indexed Birdmodel")
        #print(birdmodel)
        #print("birdmodel.pardict[do_marg]")
        
        if birdmodel.pardict["do_marg"]:
            #b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
            #b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
            #bs = [params[-3], b2, 0.0, b4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            #IF ERROR, CHECK INDEXING
            b2 = (params[5+(3*index)] + params[6+(3*index)])/np.sqrt(2.0)
            b4 = (params[5+(3*index)] - params[6+(3*index)])/np.sqrt(2.0)
            bs = [params[4 + (3*index)], b2, 0.0, b4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
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
        P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, data.data["x_data"])
        Pi = birdmodel.get_Pi_for_marg(Ploop, bs[0], data.data["shot_noise"], data.data["x_data"])

        chi_squared = chi_squared + birdmodel.compute_chi2(P_model_interp, Pi, data.data)
        #print("In-loop chi-squared %lf" %birdmodel.compute_chi2(P_model_interp, Pi, data.data))

        #update_plot(pardicts, fittingdata.data["x_data"], P_model_interp, plt)
        #if np.random.rand() < 0.1:
        #    print(params, chi_squared)

    print("Returned chi_squared = %lf" %chi_squared)
    return -0.5 * chi_squared


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    #WITHOUT WINDOW FUNCTIONS
    #configfile_BOSS_NGC_z1 = "../config/tbird_UNIT_simfits_BOSS_NGClowz.txt"
    #configfile_BOSS_NGC_z3 = "../config/tbird_UNIT_simfits_BOSS_NGChighz.txt"
    #WITH WINDOW FUNCTIONS
    #configfile_BOSS_NGC_z1 = "../config/tbird_BOSS_NGC_z1_WinFunc.txt"
    #configfile_BOSS_NGC_z3 = "../config/tbird_BOSS_NGC_z3_WinFunc.txt"
    #Test- Passing one func without WinFunc, and one with Beutler (To test whether Beutler Win Func give us a reasonable answer)
    #configfile_BOSS_NGC_z1 = "../config/tbird_BOSS_NGC_z1_WinFunc.txt"
    #configfile_BOSS_NGC_z3 = "../config/tbird_BOSS_NGC_z3_BeutlerWinFuncTest.txt"
    #Test- NGC z1 + z3 with window functions that normalise using monopole points AFTER s = 0 
    configfile_BOSS_NGC_z1 = "../config/tbird_BOSS_NGC_z1_WinFunc_s1fixed.txt"
    configfile_BOSS_NGC_z3 = "../config/tbird_BOSS_NGC_z3_WinFunc_s1fixed.txt"
    pardict_BOSS_NGC_z1 = ConfigObj(configfile_BOSS_NGC_z1)
    print(pardict_BOSS_NGC_z1)
    pardict_BOSS_NGC_z3 = ConfigObj(configfile_BOSS_NGC_z3)

    n_datasets = 2 #Used when assigning walkers in emcee

#LIST EVERY INDIVIDUAL PARDICT

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict_BOSS_NGC_z1 = format_pardict(pardict_BOSS_NGC_z1)
    pardict_BOSS_NGC_z3 = format_pardict(pardict_BOSS_NGC_z3)

    pardicts = [pardict_BOSS_NGC_z1, pardict_BOSS_NGC_z3]

#FORMAT EVERY INDIVIDUAL INI

    # Set up the data
    fittingdata_BOSS_NGC_z1 = FittingData(pardict_BOSS_NGC_z1, shot_noise=float(pardict_BOSS_NGC_z1["shot_noise"]))
    fittingdata_BOSS_NGC_z3 = FittingData(pardict_BOSS_NGC_z3, shot_noise=float(pardict_BOSS_NGC_z3["shot_noise"]))
    
    #Check to make sure everything is defined correctly:
    print("fittingdata_BOSS_NGC_z1 = ")
    print(pardict_BOSS_NGC_z1)
    print("fittingdata_BOSS_NGC_z3 = ")
    print(pardict_BOSS_NGC_z3)
    
    fittingdatas = [fittingdata_BOSS_NGC_z1, fittingdata_BOSS_NGC_z3]
    #fittingdata_BOSS_NGC_z1["fit_data"]
    #print(fittingdata_BOSS_NGC_z1.fittingdata["fit_data"])

    #index = fittingdatas.index(fittingdata_BOSS_NGC_z3)
    #print("Test to find corresponding pardict for dataset")
    #print(pardicts[index])

#CREATE LIST OF FITTING DATA

    # Set up the BirdModel
    birdmodel_BOSS_NGC_z1 = BirdModel(pardict_BOSS_NGC_z1, template=False)
    birdmodel_BOSS_NGC_z3 = BirdModel(pardict_BOSS_NGC_z3, template=False)
    
    birdmodels = [birdmodel_BOSS_NGC_z1, birdmodel_BOSS_NGC_z3]

#CREATE LIST OF BIRDMODELS FOR EACH PARDICT

    plt = None

#DO THIS FOR EVERY INDIVIDUAL BIRDMODEL
#USE FOR LOOP, DEFINE BIASES FOR EACH INDIVIDUAL (BOTH COSMO AND SKY PATCH)

    if birdmodel_BOSS_NGC_z1.pardict["do_marg"]:
        start = np.concatenate([birdmodel_BOSS_NGC_z1.valueref[:4], [1.3, 0.5, 0.5]])
    else:
        start = np.concatenate([birdmodel_BOSS_NGC_z1.valueref[:4], [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
#Note- We create a data vector which shares the 4 cosmo parameters, and defines separate nuisance parameters, which is why we define 
#it in this way
    if birdmodel_BOSS_NGC_z3.pardict["do_marg"]:
        start = np.concatenate([start, [1.3, 0.5, 0.5]])
    else:
        start = np.concatenate([start, [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

    #for i, (pardict, start, model, data) in enumerate(zip(pardicts, starts, birdmodels, fittingdatas)):
        #index = fittingdatas.index(fittingdata_BOSS_NGC_z3)
        #print("INDEX")
        #print(index)
        #print(pardicts[index])
    # Does an optimization
    result = do_optimization(lambda *args: -lnpost(*args), start, birdmodels, fittingdatas, plt)

#DO EMCEE FOR LIST OF BIRDMODELS, FITTINGDATA [BIRDMODEL1, BIRDMODEL2 ETC.]

    # Does an MCMC
    #do_emcee(lnpost, start, birdmodels, fittingdatas, plt)
