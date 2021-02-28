import numpy as np
import sys
from configobj import ConfigObj

sys.path.append("../")
from fitting_codes.fitting_utils_template import (
    FittingData,
    BirdModel,
    create_plot,
    update_plot,
    format_pardict,
    do_optimization,
)

if __name__ == "__main__":

    #params- best fit from fit_UNIT_template.
    #Using 1st iteration, which shouldn't be impacted by any errors in the template recommendation code, i.e. alpha_par, alpha_perp, fs8 from first fid should fit Pk
    #params = [ 1.03331642,  1.0393795 ,  0.4485589 ,  1.55756676,  1.23589323, -0.5144482 ] #true fits from first iteration
    #params = [ 0.93331642,  0.9393795 , 0.4485589 ,  1.55756676,  1.23589323, -0.5144482 ] #testing Pk grid follows correct direction w/ alpha
    #params = [1.03197449, 1.03754415, 0.42460016, 1.29056367, 0.36491808, 1.6574533 ] #Best fit from iteration 4
    #params = [1.03331642,  1.0393795 , 0.4485589, 1.29056367, 1.23589323, 1.6574533 ] #testing best fit from one iteration on grid of another
    #params = [1.0,  1.0 , 0.43538391468, 1.3, 0.5, 0.5] #plotting central point of each grid to show reasonable difference
    params = [1.0,  1.0 , 0.43538391468, 1.3, 0.5, 0.5] #plotting central point of each grid to show reasonable difference
    configfile = "../config/DESI_template_stress_test_iter1.ini"
    #configfile = "../config/DESI_template_stress_test_iter4.ini"  
    pardict = ConfigObj(configfile)
    pardict = format_pardict(pardict)
    print(pardict)
    fittingdata = FittingData(pardict, shot_noise=float(pardict["shot_noise"]))
    birdmodel = BirdModel(pardict, template=True)
    plt = create_plot(pardict, fittingdata)

    #-------------------------------------------------------------------------------------
    #With the models/data instantiated, and the plot object constructed, we now must
    #construct the full data vector from our marginalised output

    b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
    b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
    bs = [params[-3], b2, 0.0, b4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Get the bird model
    alpha_perp, alpha_par, fsigma8 = params[:3]
    f = fsigma8 / birdmodel.valueref[3]
    
    Plin, Ploop = birdmodel.compute_pk([alpha_perp, alpha_par, f, birdmodel.valueref[3]])
    P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
    Pi = birdmodel.get_Pi_for_marg(Ploop, bs[0], fittingdata.data["shot_noise"], fittingdata.data["x_data"])
    
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

    update_plot(pardict, fittingdata.data["x_data"], P_model_interp, plt, keep=True)

