import sys
import numpy as np
import pandas as pd
from configobj import ConfigObj
from chainconsumer import ChainConsumer

#sys.path.append("../../../")
sys.path.append("../")
from tbird.Grid import run_camb, run_class
from fitting_codes.fitting_utils import (
    read_chain_backend,
    BirdModel,
    FittingData,
    create_plot,
    update_plot,
    format_pardict,
)


if __name__ == "__main__":

    # First read in the config file and compute Da_fid, Hz_fid and sigma8_fid
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)
    pardict = format_pardict(pardict)
    if pardict["code"] == "CAMB":
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12_fid, r_d_fid = run_camb(pardict)
    else:
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12_fid, r_d_fid = run_class(pardict)

    # Set the chainfiles and names for each chain
    if pardict["do_corr"]:
        chainfiles = [
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_35hex35_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_30hex30_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_25hex25_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_20hex25_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_20hex20_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_15hex25_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_15hex20_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_15hex15_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_10hex25_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_10hex20_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_10hex15_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_10hex10_2order_hex_marg.hdf5",
        ]
        figfile = [
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_2order_hex_marg.pdf",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_2order_hex_marg_1d.pdf",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_2order_hex_marg_summary.pdf",
        ]
        names = [
            r"$s_{\mathrm{min}}=35\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{35}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=30\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{30}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=25\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{25}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=20\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{25}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=20\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{20}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=15\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{25}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=15\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{20}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=15\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{15}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=10\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{25}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=10\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{20}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=10\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{15}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=10\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{10}\,h\mathrm{Mpc^{-1}}$",
        ]
    else:
        chainfiles = [
            # "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.10hex0.10_2order_hex_marg.hdf5",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.15hex0.15_2order_hex_marg.hdf5",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.20hex0.15_2order_hex_marg.hdf5",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.20hex0.20_2order_hex_marg.hdf5",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.25hex0.15_2order_hex_marg.hdf5",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.25hex0.20_2order_hex_marg.hdf5",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.25hex0.25_2order_hex_marg.hdf5",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.30hex0.15_2order_hex_marg.hdf5",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.30hex0.20_2order_hex_marg.hdf5",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.30hex0.25_2order_hex_marg.hdf5",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.30hex0.30_2order_hex_marg.hdf5",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.40hex0.15_2order_hex_marg.hdf5",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.40hex0.20_2order_hex_marg.hdf5",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.40hex0.25_2order_hex_marg.hdf5",
            # "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_0.40hex0.40_2order_hex_marg.hdf5",
             #"/mnt/c/Users/s4393083/pybird/output_files/SimFits/chain_BOSS_NGClowz/chain_BOSS_NGClowz_pk_0.15hex0.15_2order_hex_marg.hdf5",
            
             
            #"stored_chain_Cooke19OBPrior_BOSS_NGClowz_pk_0.25hex0.15_2order_hex_marg.hdf5",
            # "chain_BBNOBPrior_BOSS_NGChighz_UpdatedCovMatrix__pk_0.25hex0.15_2order_hex_marg.hdf5",
            # "TestBOSSNGC_z1_z3_CombinedFit_Converged.hdf5",
            #"tbird_BOSS_NGC_z1_WinFunc_s1fixed__pk_0.25hex0.15_2order_hex_marg.hdf5",
            #"tbird_BOSS_NGC_z3_BeutlerWinFuncTest_FixedObPrior__pk_0.25hex0.15_2order_hex_marg.hdf5",
            #"TestBOSSNGC_z1_z3_CombinedFit_WinFunc_s1fixed_40000Steps.hdf5",
            "NGC_CMASS_DAmico2019_FixedOBRatio_LinearBBNPrior_FixedPk_pk_0.20hex0.20_3order_hex_marg.hdf5"
            ]
        figfile = [
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_2order_hex_marg.pdf",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_2order_hex_marg_1d.pdf",
            #"/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_smallgrid_BBNprior_pk_2order_hex_marg_summary.pdf",
            #"/mnt/c/Users/s4393083/pybird/output_files/chain_BOSS_NGClowz_pk_0.15hex0.15_2order_hex_marg.pdf",
            "DAmico_3rd_Taylor_order_fixed_O_b_ratio_No_O_b_prior_hexadecapole_FixedPk.pdf",
            #"/mnt/c/Users/s4393083/pybird/output_files/SimFit_chain_marg_1d_test_pk0.15_hex0.15.pdf",
            "DAmico_3rd_Taylor_order_fixed_O_b_ratio_No_O_b_prior_hexadecapole_FixedPk.pdf",
            #"/mnt/c/Users/s4393083/pybird/output_files/SimFit_chain_marg_summary_test.pdf",
        ]
        names = [
            # r"$k_{\mathrm{max}}=0.10\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.10}\,h\mathrm{Mpc^{-1}}$",
            #r"$k_{\mathrm{max}}=0.15\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.15}\,h\mathrm{Mpc^{-1}}$",
            #r"$k_{\mathrm{max}}=0.20\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.15}\,h\mathrm{Mpc^{-1}}$",
            #r"$k_{\mathrm{max}}=0.20\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.20}\,h\mathrm{Mpc^{-1}}$",
            #r"$k_{\mathrm{max}}=0.25\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.15}\,h\mathrm{Mpc^{-1}}$",
            #r"$k_{\mathrm{max}}=0.25\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.20}\,h\mathrm{Mpc^{-1}}$",
            #r"$k_{\mathrm{max}}=0.25\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.25}\,h\mathrm{Mpc^{-1}}$",
            #r"$k_{\mathrm{max}}=0.30\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.15}\,h\mathrm{Mpc^{-1}}$",
            #r"$k_{\mathrm{max}}=0.30\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.20}\,h\mathrm{Mpc^{-1}}$",
            #r"$k_{\mathrm{max}}=0.30\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.25}\,h\mathrm{Mpc^{-1}}$",
            #r"$k_{\mathrm{max}}=0.30\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.30}\,h\mathrm{Mpc^{-1}}$",
            #r"$k_{\mathrm{max}}=0.40\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.15}\,h\mathrm{Mpc^{-1}}$",
            #r"$k_{\mathrm{max}}=0.40\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.20}\,h\mathrm{Mpc^{-1}}$",
            #r"$k_{\mathrm{max}}=0.15\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.15}\,h\mathrm{Mpc^{-1}}$",
            r"$\rm{NGC \ CMASS \ Expanded \ Grid, \ fixed \ OB, No BBN, \ Hexadecapole, \ 3^{rd} \ Order}$",
            r"NGC highz w/ Beutler Win Func $k_{\mathrm{max}}=0.25\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.15}\,h\mathrm{Mpc^{-1}}$",
            r"Simultaneous lowz-highz Post WinFunc $k_{\mathrm{max}}=0.25\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.15}\,h\mathrm{Mpc^{-1}}$",
        ]


    #truths from config file:
    #truths = {
    #    r"$A_{s}\times 10^{9}$": np.exp(float(pardict["ln10^{10}A_s"])) / 1.0e1,
    #    r"$h$": float(pardict["h"]),
    #    r"$\Omega_{m}$": Om_fid,
    #    r"$\omega_{cdm}$": float(pardict["omega_cdm"]),
    #    r"$\omega_{b}$": float(pardict["omega_b"]),
    #}
    
    #truths (from D'Amico 2019)
    truths = {
        r"$ln(10^{10} A_s)$": 2.6,
        r"$h$": 0.722,
        r"$\Omega_{m}$": 0.309,
        #r"$\omega_{cdm}$": float(pardict["omega_cdm"]), #omega_cdm and omega_b not given in D'Amico 2019, use truth from config file for these
        #r"$\omega_{b}$": float(pardict["omega_b"]),
        }

    # Output name for the figure

    c = ChainConsumer()

    bestfits = []
    for chaini, chainfile in enumerate(chainfiles):

        #For Omega_b/omega_cdm constraints:    
        '''
        burntin, bestfit, like = read_chain_backend(chainfile)
        #burntin[:, 0] = np.exp(burntin[:, 0]) / 1.0e1 #convert ln10^10(as) to As*10^9
        #omega_b = float(pardict["omega_b"]) / float(pardict["omega_cdm"]) * burntin[:, 2] 
        omega_b = burntin[:, 3] #Original setting, provided Omega_m = 4
        print(omega_b)
        #Omega_m = (burntin[:, 2] + omega_b + (0.06 / 93.14)) / burntin[:, 1] ** 2 #Is this a BBN constraint?
        Omega_m = (burntin[:, 2] + burntin[: ,3])/(burntin[:, 1]**2) #Check formatting
        paramnames = [r"$ln(10^{10} A_s)$", r"$h$", r"$\omega_{cdm}$", r"$\omega_{b}$", r"$\Omega_{m}$"]
        c.add_chain(
            np.hstack([burntin[:, :4], Omega_m[:, None]]), parameters=paramnames, name=names[chaini], posterior=like #Original, providing \omega_b = 1.9?
            #np.hstack([burntin[:, :3], omega_b[:, None], Omega_m[:, None]]), parameters=paramnames, name=names[chaini], posterior=like
        )
        bestfits.append(bestfit)

    print(bestfits)
    #extents = [(1.8, 2.6), (0.66, 0.698), (0.108, 0.132), (0.29, 0.33)]
    extents = [(2, 4), (0.5, 0.9), (0.10, 0.15), (0.0, 0.1), (0.22, 0.38)] #updated range for D'Amico Comparison
    #extents = [(1.65, 2.2), (0.66, 0.698), (0.108, 0.132), (1.7, 1.9), (0.29, 0.33)]
    c.configure(bar_shade=True)
    #fig = c.plotter.plot_summary(filename=figfile[1], truth=truths, display=False, extents=extents)
    #print(c.analysis.get_summary())
    fig = c.plotter.plot(filename=figfile[1], truth=truths, display=False, extents=extents)
          '''
          
          #For Omega_m constraints
        burntin, bestfit, like = read_chain_backend(chainfile)
        #burntin[:, 0] = np.exp(burntin[:, 0]) / 1.0e1 #convert ln10^10(as) to As*10^9
        #omega_b = float(pardict["omega_b"]) / float(pardict["omega_cdm"]) * burntin[:, 2] 
        #omega_b = (burntin[:, 3]*0.155)/(burntin[:, 2]**2) #Original setting, provided Omega_m = 4
        #print(omega_b)
        #omega_cdm = (burntin[:, 3]*0.845)/(burntin[:, 2]**2)
        #Omega_m = (burntin[:, 2] + omega_b + (0.06 / 93.14)) / burntin[:, 1] ** 2 #Is this a BBN constraint?
        #Omega_m = (burntin[:, 2] + burntin[: ,3])/(burntin[:, 1]**2) #Check formatting
        paramnames = [r"$ln(10^{10} A_s)$", r"$h$", r"$\Omega_{m}$"]
        c.add_chain(
            np.hstack([burntin[:, :3]]), parameters=paramnames, name=names[chaini], posterior=like #Original, providing \omega_b = 1.9?
            #np.hstack([burntin[:, :2], Omega_m[:, None]]), parameters=paramnames, name=names[chaini], posterior=like
        )
        bestfits.append(bestfit)

    print(bestfits)
    #extents = [(1.8, 2.6), (0.66, 0.698), (0.108, 0.132), (0.29, 0.33)]
    extents = [(2, 4), (0.5, 0.9), (0.2, 0.4), (0.0, 0.3), (0.0, 0.05)] #updated range for D'Amico Comparison
    #extents = [(1.65, 2.2), (0.66, 0.698), (0.108, 0.132), (1.7, 1.9), (0.29, 0.33)]
    c.configure(bar_shade=True)
    #fig = c.plotter.plot_summary(filename=figfile[1], truth=truths, display=False, extents=extents)
    #print(c.analysis.get_summary())
    fig = c.plotter.plot(filename=figfile[1], truth=truths, display=False, extents=extents)

    # Get the bestfit bird model
    if False:

        import matplotlib.pyplot as plt

        fig = plt.figure()
        params = bestfits[1]
        fittingdata = FittingData(pardict, shot_noise=float(pardict["shot_noise"]))

        # Set up the BirdModel
        birdmodel = BirdModel(pardict)
        print(birdmodel.valueref)

        # Plotting (for checking/debugging, should turn off for production runs)
        plt = create_plot(pardict, fittingdata)

        if birdmodel.pardict["do_marg"]:
            b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
            b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
            bs = [params[-3], b2, 0.0, b4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            b2 = (params[-9] + params[-7]) / np.sqrt(2.0)
            b4 = (params[-9] - params[-7]) / np.sqrt(2.0)
            bs = [
                params[-10],
                b2,
                params[-8],
                b4,
                params[-6],
                params[-5],
                params[-4],
                params[-3] * fittingdata.data["shot_noise"],
                params[-2] * fittingdata.data["shot_noise"],
                params[-1] * fittingdata.data["shot_noise"],
            ]

        ln10As, h, omega_cdm, omega_b = params[:4]
        #omega_b = birdmodel.valueref[3]

        Plin, Ploop = birdmodel.compute_smallgrid_BBNprior_pk([ln10As, h, omega_cdm, omega_b])
        P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
        Pi = birdmodel.get_Pi_for_marg(Ploop, bs[0], fittingdata.data["shot_noise"], fittingdata.data["x_data"])
        chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)

        if birdmodel.pardict["do_marg"]:
            bs_analytic = birdmodel.compute_bestfit_analytic(Pi, fittingdata.data)
            print(bs_analytic)
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
                bs_analytic[4],
                bs_analytic[5],
                bs_analytic[6],
            ]
            P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
            chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)
        print(params, chi_squared)

        update_plot(pardict, fittingdata.data["x_data"], P_model_interp, plt, keep=True)

        """np.savetxt(
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_xi_30-200_varyh_nohex_all_bestfit.dat",
            np.c_[
                fittingdata.data["x_data"],
                P_model[: len(fittingdata.data["x_data"])],
                P_model[len(fittingdata.data["x_data"]) :],
            ],
            header="k       P0          P2",
        )"""
