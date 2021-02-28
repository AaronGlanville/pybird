import numpy as np
import sys
from configobj import ConfigObj

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


#NGC z3
#params = [2.65974009, 0.68944003, 0.12377864, 0.02166142, 2.05352652, 0.35634227, 0.17404805]

#SGC z3
#params = [2.68348093,  0.66975773,  0.12214865,  0.02165877,  2.3394121, 0.6986747 , -0.0331882]

#SGC z1 + NGC z1
#params = [3.08166488, 0.6627053, 0.11203482, 0.02166503, 1.80325958, 0.55003127, 0.23220423, 1.82812768, 0.80374831, 0.22364952]

#SGC z3 + NGC z3
#params = [2.62857279,  0.68501402,  0.12705627,  0.02165538,  2.36831385, 0.50491058,  0.12163542,  2.35195127,  0.74271493, -0.05290764]

#SGC z3 + NGC z3 with updated shot_noise
params = [ 2.63789963,  0.68524438,  0.12702671,  0.02165575,  2.3599131 , 0.55354874,  0.16638361,  2.34068959,  0.75628652, -0.0178072 ]

if __name__ == "__main__":

    NGC_configfile = "../config/tbird_NGC_z3_s10fixed_singlefit.txt"
    SGC_configfile = "../config/tbird_SGC_z3_s10fixed_singlefit.txt"
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
    colours = ["b", "r", "g", "c", "n"]
    plt = create_plot_combined(pardict_NGC, pardict_SGC, colours, fittingdata_combined) #NOTE: Pardict NGC must contain both gridnames for the NGC and SGC sections

    #-------------------------------------------------------------------------------------
    #With the models/data instantiated, and the plot object constructed, we now must
    #construct the full data vector from our marginalised output

    b2_SGC = (params[-2] + params[-1]) / np.sqrt(2.0)
    b4_SGC = (params[-2] - params[-1]) / np.sqrt(2.0)
    bs_SGC = [params[-3], b2_SGC, 0.0, b4_SGC, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


    b2_NGC = (params[-5] + params[-4]) / np.sqrt(2.0)
    b4_NGC = (params[-5] - params[-4]) / np.sqrt(2.0)
    bs_NGC = [params[-6], b2_NGC, 0.0, b4_NGC, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    ln10As, h, omega_cdm, omega_b = params[:4]
    Plin_NGC, Ploop_NGC, Plin_SGC, Ploop_SGC = birdmodel_combined.compute_pk_separategrids([ln10As, h, omega_cdm, omega_b])
    Pi_NGC, Pi_SGC, Pi = birdmodel_combined.get_Pi_for_marg_separategrids(Ploop_NGC, Ploop_SGC, bs_NGC[0], bs_SGC[0], NGC_shot_noise, SGC_shot_noise, fittingdata_combined.data["x_data"])
    P_model, P_model_interp, P_model_NGC, P_model_interp_NGC, P_model_SGC, P_model_interp_SGC = birdmodel_combined.compute_model_separategrids(bs_NGC, bs_SGC, Plin_NGC, Ploop_NGC, Plin_SGC, Ploop_SGC, fittingdata_combined.data["x_data"])
    bs_analytic = birdmodel_combined.compute_bestfit_analytic(Pi, fittingdata_combined.data, P_model_interp) #Should output NGC-SGC bias params in one object
            

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
    P_model, P_model_interp, P_model_NGC, P_model_interp_NGC, P_model_SGC, P_model_interp_SGC = birdmodel_combined.compute_model_separategrids(bs_NGC, bs_SGC, Plin_NGC, Ploop_NGC, Plin_SGC, Ploop_SGC, fittingdata_combined.data["x_data"])
    update_plot_combined(pardict_NGC, fittingdata_combined.data["x_data"], P_model_interp_NGC, P_model_interp_SGC, colours, "NGC+SGC_z3_combined_edited", plt, keep=True)
