#Fitting for correlated data sets (e.g. NGC/SGC slices at constant
#effective redshift). We want to pass a single data vector
#containing P0_NGC, P2_NGC, P0_SGC, P2_SGC (with corresponding model)
#However define independent bias parameters (due to selection systematics) 

import numpy as np
import sys
from configobj import ConfigObj
import os
#import pybird

sys.path.append("../")

from fitting_codes.fitting_utils_NGCSGC_combined import (
    FittingData_NGCSGC,
    BirdModel,
    create_plot_combined,
    update_plot,
    format_pardict,
    do_optimization,
)

def do_emcee(func, start, birdmodel, fittingdata, plt):

    import emcee
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
    
    #chainfile = 'TestBOSSNGC_z1NGC+SGC_CombinedFit_WinFunc_s10fixed_40000Steps.hdf5'
    #chainfile = 'z1_NGC+NGC_40000_Stept_Test.hdf5'
    chainfile = 'z1_NGC+NGC_40000_steps_Test_UpdatedPi.hdf5'
    print(chainfile)

    # Set up the backend
    backend = emcee.backends.HDFBackend(chainfile)
    backend.reset(nwalkers, nparams)
    # Initialize the sampler
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
    if birdmodel.pardict["do_marg"]:
        b1_NGC, c2_NGC, c4_NGC = params[4:7]
        b1_SGC, c2_SGC, c4_SGC = params[7:10]
    else:
        b1_NGC, c2_NGC, b3_NGC, c4_NGC, cct_NGC, cr1_NGC, cr2_NGC, ce1_NGC, cemono_NGC, cequad_NGC, bnlo_NGC, b1_SGC, c2_SGC, b3_SGC, c4_SGC, cct_SGC, cr1_SGC, cr2_SGC, ce1_SGC, cemono_SGC, cequad_SGC, bnlo_SGC = params[-22:]
        
    ln10As, h, omega_cdm, omega_b = params[:4]

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
    b1_NGC_min = birdmodel.pardict["b1_min"]
    b1_SGC_min = birdmodel.pardict["b1_min"]
    b1_NGC_max = birdmodel.pardict["b1_max"]
    b1_SGC_max = birdmodel.pardict["b1_max"]
    c2_NGC_min = birdmodel.pardict["c2_min"]
    c2_SGC_min = birdmodel.pardict["c2_min"]
    c2_NGC_max = birdmodel.pardict["c2_max"]
    c2_SGC_max = birdmodel.pardict["c2_max"]
    c4_NGC_width = birdmodel.pardict["c4_width"]
    c4_SGC_width = birdmodel.pardict["c4_width"]
    # Flat prior for b1
    if b1_NGC < float(b1_NGC_min) or b1_NGC > float(b1_NGC_max):
        return -np.inf
    if b1_SGC < float(b1_SGC_min) or b1_SGC > float(b1_SGC_max):
        return -np.inf
    # Flat prior for c2
    if c2_NGC < float(c2_NGC_min) or c2_NGC > float(c2_NGC_max):
        return -np.inf
    if c2_SGC < float(c2_SGC_min) or c2_SGC > float(c2_SGC_max):
        return -np.inf

    # Gaussian prior for c4 =
    c4_NGC_prior = -0.5 * float(c4_NGC_width)* c4_NGC ** 2
    c4_SGC_prior = -0.5 * float(c4_SGC_width)* c4_SGC ** 2
    
    if birdmodel.pardict["do_marg"]:
        return omega_b_prior + c4_NGC_prior + c4_SGC_prior #Not sure if they should just be added?

    else:

#DEFINE PRIOR LIST IN EACH INI
        
        # Gaussian prior for b3 of width 2 centred on 0
        b3_prior = -0.5 * ((0.25 * b3_NGC ** 2)+(0.25 * b3_SGC ** 2))

        # Gaussian prior for cct of width 2 centred on 0
        cct_prior = -0.5 * ((0.25 * cct_NGC ** 2)+(0.25 * cct_SGC ** 2))

        # Gaussian prior for cr1 of width 4 centred on 0
        cr1_prior = -0.5 * ((0.0625 * cr1_NGC ** 2)+(0.0625 * cr1_SGC ** 2))

        # Gaussian prior for cr1 of width 4 centred on 0
        cr2_prior = -0.5 * ((0.0625 * cr2_NGC ** 2)+(0.0625 * cr2_SGC ** 2))

        # Gaussian prior for ce1 of width 2 centred on 0
        ce1_prior = -0.5 * ((0.25 * ce1_NGC ** 2)+(0.25 * ce1_SGC ** 2))

        # Gaussian prior for cemono of width 2 centred on 0
        cemono_prior = -0.5 * ((0.25 * cemono_NGC ** 2)+(0.25 * cemono_SGC ** 2))

        # Gaussian prior for cequad of width 2 centred on 0
        cequad_prior = -0.5 * ((0.25 * cequad_NGC ** 2)+(0.25 * cequad_SGC ** 2))

        # Gaussian prior for bnlo of width 2 centred on 0
        bnlo_prior = -0.5 * ((0.25 * bnlo_NGC ** 2)+(0.25 * bnlo_NGC ** 2))

        return ( #Only for non-marginalised evaluations, not worrying for now
            omega_b_prior
            + c4_NGC_prior
            + c4_SGC_prior
            + b3_prior*2
            + cct_prior*2
            + cr1_prior*2
            + cr2_prior*2
            + ce1_prior*2
            + cemono_prior*2
            + cequad_prior*2
            + bnlo_prior*2
        )

def lnlike(params, birdmodel, fittingdata, plt):
    
    if birdmodel.pardict["do_marg"]:
        b2_NGC = (params[-5] + params[-4])/np.sqrt(2.0)
        b4_NGC = (params[-5] - params[-4])/np.sqrt(2.0)
        
        b2_SGC = (params[-2] + params[-1]) / np.sqrt(2.0)
        b4_SGC = (params[-2] - params[-1]) / np.sqrt(2.0) #check indexing if any issues come up
        
        bs_NGC = [params[-6], b2_NGC, 0.0, b4_NGC, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        bs_SGC = [params[-3], b2_SGC, 0.0, b4_SGC, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
   
    else: #haven't updated bias parameters for non-marginalised cases
    
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

    #Get the bird model
    ln10As, h, omega_cdm, omega_b = params[:4]
    Plin, Ploop = birdmodel.compute_pk([ln10As, h, omega_cdm, omega_b]) #Plin/Ploop only relies on cosmology, does not need NGC/SGC distinction
    P_model, P_model_interp = birdmodel.compute_model_NGCSGC(bs_NGC, bs_SGC, Plin, Ploop, fittingdata_combined.data["x_data"])#just need k-space, should be the same across files
    Pi = birdmodel.get_Pi_for_marg_NGCSGC(Ploop, bs_NGC[0], bs_SGC[0], NGC_shot_noise, SGC_shot_noise, fittingdata_combined.data["x_data"]) #Pi seems kinda dodgy- interpolated P from discrete points? idk     

    #Testing to see whether plotting model with 11 linear bias  parameters analytically marginalised give reasonable plots
    #NOTE: DO NOT PASS THESE TO CHI^2, FOR PLOTS ONLY
    if birdmodel.pardict["do_marg"]:
        bs_analytic = birdmodel.compute_bestfit_analytic(Pi, fittingdata_combined.data)
        #print("bs_analytic")
        #print(bs_analytic)
        bs_NGC_analytic = [
            params[-6],
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

        bs_SGC_analytic = [
            params[-3],
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

    chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata_combined.data) #data.data only used for inverse covariance matrix, joint cov_matrix for NGC/SGC
    #update_plot(pardict_BOSS_NGC_z1, fittingdata.data["x_data"], P_model_interp_plotting, plt) #Passing data vector with analytically marginalised linear bias params for plotting
    return -0.5 * chi_squared


if __name__ == "__main__":

    #CURRENT STRATEGY: Have the code read in multiple config files/pardicts etc. and use both of them
    #as inputs into our code. In future, want to move them into one config file with separate 
    #NGC/SGC data files, shot noise, and different bias params defined in code.    

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file and plot flag
    
    plot_flag = int(sys.argv[1])
    
    configfile_BOSS_NGC_z1 = "../config/tbird_NGC_z1.txt"
    #configfile_BOSS_SGC_z1 = "../config/tbird_SGC_z1.txt" #For joing NGC/SGC z1 fits
    configfile_BOSS_SGC_z1 = "../config/tbird_SGC_z1.txt" #For joing NGC/SGC z1 fits
    pardict_BOSS_NGC_z1 = ConfigObj(configfile_BOSS_NGC_z1)
    print(pardict_BOSS_NGC_z1)
    pardict_BOSS_SGC_z1 = ConfigObj(configfile_BOSS_SGC_z1)
    n_datasets = 2 #Used when assigning walkers in emcee

#LIST EVERY INDIVIDUAL PARDICT

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict_BOSS_NGC_z1 = format_pardict(pardict_BOSS_NGC_z1)
    pardict_BOSS_SGC_z1 = format_pardict(pardict_BOSS_SGC_z1)
    
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
    print(fittingdata_combined)
#CREATE LIST OF FITTING DATA

    # Set up the BirdModel
    birdmodel_combined = BirdModel(pardict_BOSS_NGC_z1, template=False) #just sets up the EFT model, does not need to be duplicated for NGC/SGC cuts
    #birdmodel_BOSS_SGC_z1 = BirdModel(pardict_BOSS_SGC_z1, template=False)
    
    #birdmodels = [birdmodel_BOSS_NGC_z1, birdmodel_BOSS_SGC_z1]

#CREATE LIST OF BIRDMODELS FOR EACH PARDICT

    plt = None
    if plot_flag:
        print("Plot_flag accepted!")
        plt = create_plot_combined(pardict_BOSS_NGC_z1, pardict_BOSS_SGC_z1, fittingdata_combined)

#DO THIS FOR EVERY INDIVIDUAL BIRDMODEL
#USE FOR LOOP, DEFINE BIASES FOR EACH INDIVIDUAL (BOTH COSMO AND SKY PATCH)

    if birdmodel_combined.pardict["do_marg"]:
        start = np.concatenate([birdmodel_combined.valueref[:4], [1.3, 0.5, 0.5, 1.3, 0.5, 0.5]])
    else:
        start = np.concatenate([birdmodel_combined.valueref[:4], [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

    # Does an optimization
    result = do_optimization(lambda *args: -lnpost(*args), start, birdmodel_combined, fittingdata_combined, plt)
    # Does an MCMC
    #do_emcee(lnpost, start, birdmodel_combined, fittingdata_combined, plt)
