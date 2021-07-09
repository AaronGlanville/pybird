import os
import numpy as np
#from montepython.likelihood_class import Likelihood
from montepython.likelihood_class import Likelihood

#Pybird stuff:
import sys
from configobj import ConfigObj
from multiprocessing import Pool

sys.path.append("../../../../")
from fitting_codes.fitting_utils_z1_z2_z3 import (
    FittingData,
    BirdModel,
    create_plot,
    update_plot,
    format_pardict,
    do_optimization,
    get_Planck,
)

#This will be the code to do all I/O before passing to generic likelihood class

#starting with generic likelihood?
#class pybird_z1_z2_z3(Likelihood):
class pybird_z1_z2_z3(Likelihood):
    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        pardict = ConfigObj(self.configfile)
        
        # Just converts strings in pardicts to numbers in int/float etc.
        self.pardict = format_pardict(pardict)

        # Set up the data
        self.fittingdata = FittingData(self.pardict)

        # Set up the BirdModels
        self.birdmodels = []
        for i in range(len(pardict["z_pk"])):
            self.birdmodels.append(BirdModel(self.pardict, direct=True, redindex=i, window=self.fittingdata.data["windows"][i]))
            #birdmodels.append(BirdModel(pardict, redindex=i))
        
        #print("fittingdata = ")
        #print(self.fittingdata)
        
        
        
    
    def loglkl(self, cosmo, data):
        
        #I think data stores the param values passed from montepython
        #and we can store pk/cov_matrix in fittingdata
        #parameters indexed as ln10As, h, omega_cdm, omega_b, [b1, c2], [b1, c2],...
        
        #start by accessing cosmological parameters
        
        #++++++++++++++++++++++++++++++
        #Added for individual stuff#
        #oned_flag=0
        #if params.ndim==1:
        #    params = params.reshape((-1, len(params)))
        oned_flag = 1
        #params = params.T
        #++++++++++++++++++++++++++++    
        
        param_names = data.get_mcmc_parameters(["varying"]) #returns string containing cosmo params ['h', 'omega_b',..] etc.
        
        ln10As = [data.mcmc_parameters[param_names[0]]["current"]]
        h = [data.mcmc_parameters[param_names[1]]["current"]]
        omega_cdm = [data.mcmc_parameters[param_names[2]]["current"]]
        omega_b = [data.mcmc_parameters[param_names[3]]["current"] * 0.01] #class takes in omega_bh^2 * 100
        
        print("Attempting to access cosmo parameters")
        print(ln10As, h, omega_cdm, omega_b)

        Picount = 0
        P_models, Plins, Ploops = [], [], []
        nmarg = len(self.birdmodels[0].eft_priors)
        nz = len(self.birdmodels[0].pardict["z_pk"])
        Pi_full = np.zeros((nz * len(self.birdmodels[0].eft_priors), len(self.fittingdata.data["fit_data"]), len([ln10As])))
        for i in range(nz):
            if self.birdmodels[0].pardict["do_marg"]:
                counter = -2 * (nz - i)
                #b2 = (params[counter + 1]) / np.sqrt(2.0)
                #b4 = (params[counter + 1]) / np.sqrt(2.0)
                
                b2 = (data.mcmc_parameters[param_names[counter+1]]["current"]) / np.sqrt(2.0)
                b4 = (data.mcmc_parameters[param_names[counter+1]]["current"]) / np.sqrt(2.0)

                margb = np.zeros(np.shape(data.mcmc_parameters[param_names[1]]["current"]))
                #print("margb (compare to margb from direct pybird)")
                #print(margb)
                bs = np.array(
                    [
                        [data.mcmc_parameters[param_names[counter]]["current"]],
                        [b2],
                        [margb],
                        [b4], 
                        [margb],
                        [margb],
                        [margb],
                        [margb],
                        [margb],
                        [margb],
                        # margb,
                    ]
                )
            else:
                print("Do_marg set to 0?")
#                counter = -11 * (nz - i)
#                b2 = (params[counter + 1] + params[counter + 3]) / np.sqrt(2.0)
#                b4 = (params[counter + 1] - params[counter + 3]) / np.sqrt(2.0)
#                bs = np.array(
#                    [
#                        params[counter],
#                        b2,
#                        params[counter + 2],
#                        b4,
#                        params[counter + 4],
#                        params[counter + 5],
#                        params[counter + 6],
#                        params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
#                        params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
#                        params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
#                        # params[counter + 10],
#                    ]
#                )
#
            #print("(np.array([ln10As, h, omega_cdm, omega_b])")
            #print(np.array([ln10As, h, omega_cdm, omega_b]))  

            Plin, Ploop = self.birdmodels[i].compute_pk(np.array([ln10As, h, omega_cdm, omega_b])) #no omega_k for now
            P_model, P_model_interp = self.birdmodels[i].compute_model(bs, Plin, Ploop, self.fittingdata.data["x_data"][i])
            Pi = self.birdmodels[i].get_Pi_for_marg(
                Ploop, bs[0], float(self.fittingdata.data["shot_noise"][i]), self.fittingdata.data["x_data"][i]
            )
            Plins.append(Plin)
            Ploops.append(Ploop)
            P_models.append(P_model_interp)
            Pi_full[i * nmarg : (i + 1) * nmarg, Picount : Picount + self.fittingdata.data["ndata"][i]] = Pi
            Picount += self.fittingdata.data["ndata"][i]

        P_model = np.concatenate(P_models)
        # chi_squared = birdmodels[0].compute_chi2(P_model, Pi_full, fittingdata.data)

        # Now get the best-fit values for parameters we don't care about
        P_models = []
        bs_analytic = self.birdmodels[0].compute_bestfit_analytic(Pi_full[:, :, 0], self.fittingdata.data, P_model[:, 0])
        self.pardict["do_marg"] = 0
        for i in range(nz):
            #original configuration
            #counter = -2 * (nz - i)
            #b2 = (params[counter + 1, 0]) / np.sqrt(2.0)
            #b4 = (params[counter + 1, 0]) / np.sqrt(2.0)
            
            #my attempt:
            counter = -2 * (nz - i)
            b2 = (data.mcmc_parameters[param_names[counter+1]]["current"]) / np.sqrt(2.0)
            b4 = (data.mcmc_parameters[param_names[counter+1]]["current"]) / np.sqrt(2.0)
            
            bs = np.array(
                [
                    data.mcmc_parameters[param_names[counter]]["current"],
                    b2,
                    bs_analytic[7 * i],
                    b4,
                    bs_analytic[7 * i + 1],
                    bs_analytic[7 * i + 2],
                    bs_analytic[7 * i + 3],
                    bs_analytic[7 * i + 4] * float(self.fittingdata.data["shot_noise"][i]),
                    bs_analytic[7 * i + 5] * float(self.fittingdata.data["shot_noise"][i]),
                    bs_analytic[7 * i + 6] * float(self.fittingdata.data["shot_noise"][i]),
                    # bs_analytic[8 * i + 7],
                ]
            )[:, None]
            
            #print("bs = ")
            #print(bs)
            print("b1 = ")
            print(data.mcmc_parameters[param_names[counter]]["current"])
            print("------------------------")
            
            #FAILING HERE- DIMENSION ERROR?
            P_model, P_model_interp = self.birdmodels[i].compute_model(
                bs, Plins[i][:, :, :, 0, None], Ploops[i][:, :, :, 0, None], self.fittingdata.data["x_data"][i]
            )
            P_models.append(P_model_interp[:, 0])
        chi_squared_print = self.birdmodels[0].compute_chi2(np.concatenate(P_models), Pi_full[:, :, 0], self.fittingdata.data)
        self.pardict["do_marg"] = 1
        print("loglkl print = %lf" %(-0.5 * chi_squared_print))

        return -0.5 * chi_squared_print
        