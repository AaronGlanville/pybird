# This file contains useful routines that should work regardless of whether you are fitting
# with fixed or varying template, and for any number of cosmological parameters

import os
import copy
import numpy as np
import scipy as sp
from scipy.interpolate import splrep, splev
import sys
from scipy.linalg import lapack, cholesky
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

sys.path.append("../")
from pybird import pybird
from tbird.Grid import grid_properties, grid_properties_template, run_camb, run_class
from tbird.computederivs import get_grids, get_template_grids, get_PSTaylor, get_ParamsTaylor

# Wrapper around the pybird data and model evaluation
class BirdModel:
    def __init__(self, pardict, template=False, direct=False):

        self.pardict = pardict
        self.Nl = 3 if pardict["do_hex"] else 2
        self.template = template
        self.direct = direct

        # Some constants for the EFT model
        self.k_m, self.k_nl = 0.7, 0.7
        self.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0]) #standard gaussian widths for EFT model
        self.eft_priors = np.concatenate((self.eft_priors, self.eft_priors))
        self.priormat = np.diagflat(1.0 / self.eft_priors ** 2)
        #print("priormat = ")
        #print(self.priormat)
        # Get some values at the grid centre
        if pardict["code"] == "CAMB":
            self.kmod, self.Pmod, self.Om, self.Da, self.Hz, self.fN, self.sigma8, self.sigma12, self.r_d = run_camb(
                pardict
            )
        else:
            self.kmod, self.Pmod, self.Om, self.Da, self.Hz, self.fN, self.sigma8, self.sigma12, self.r_d = run_class(
                pardict
            )

        if self.template:
            self.valueref, self.delta, self.flattenedgrid, self.truecrd = grid_properties_template(
                pardict, self.fN, self.sigma8
            )
        else:
            self.valueref, self.delta, self.flattenedgrid, self.truecrd = grid_properties(pardict)

        # Prepare the model
        if self.direct:
            # print("Direct not currently supported :(")
            # exit()
            if self.template:
                self.correlator = self.setup_pybird()
                self.kin = self.correlator.projection.xout
            else:
                self.correlator = self.setup_pybird()
                self.kin = self.correlator.projection.xout
        else:
            #self.kin, self.paramsmod, self.linmod, self.loopmod = self.load_model()
            self.kin, self.paramsmod_NGC, self.linmod_NGC, self.loopmod_NGC, self.paramsmod_SGC, self.linmod_SGC, self.loopmod_SGC = self.load_model_NGC_SGC()

    def setup_pybird(self):

        from pybird_dev.pybird import Correlator

        Nl = 3 if self.pardict["do_hex"] else 2
        optiresum = True if self.pardict["do_corr"] else False
        output = "bCf" if self.pardict["do_corr"] else "bPk"
        z_pk = float(self.pardict["z_pk"])
        kmax = None if self.pardict["do_corr"] else 0.5
        correlator = Correlator()

        # Set up pybird
        correlator.set(
            {
                "output": output,
                "multipole": Nl,
                "z": float(self.pardict["z_pk"]),
                "optiresum": optiresum,
                "with_bias": False,
                "with_nlo_bias": True,
                "with_exact_time": True,
                "with_AP": True,
                "kmax": kmax,
                "DA_AP": self.Da,
                "H_AP": self.Hz,
            }
        )

        return correlator

    def load_model_NGC_SGC(self): #+
        kin_NGC, paramsmod_NGC, linmod_NGC, loopmod_NGC = self.load_model_individual(r"gridname_NGC", r"outgrid_NGC") #+
        kin_SGC, paramsmod_SGC, linmod_SGC, loopmod_SGC = self.load_model_individual(r"gridname_SGC", r"outgrid_SGC") #+
        return kin_NGC, paramsmod_NGC, linmod_NGC, loopmod_NGC, paramsmod_SGC, linmod_SGC, loopmod_SGC #+

    def load_model_individual(self, gridname, outgrid):

        # Load in the model components
        gridname = self.pardict["code"].lower() + "-" + self.pardict[gridname]
        if self.pardict["taylor_order"]:
            if self.template:
                paramsmod = None
                if self.pardict["do_corr"]:
                    linmod = np.load(
                        os.path.join(self.pardict[outgrid], "DerClin_%s_template.npy" % gridname), allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(self.pardict[outgrid], "DerCloop_%s_template.npy" % gridname), allow_pickle=True,
                    )
                else:
                    linmod = np.load(
                        os.path.join(self.pardict[outgrid], "DerPlin_%s_template.npy" % gridname), allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(self.pardict[outgrid], "DerPloop_%s_template.npy" % gridname), allow_pickle=True,
                    )
            else:
                paramsmod = np.load(
                    os.path.join(self.pardict[outgrid], "DerParams_%s.npy" % gridname), allow_pickle=True,
                )
                if self.pardict["do_corr"]:
                    linmod = np.load(
                        os.path.join(self.pardict[outgrid], "DerClin_%s.npy" % gridname), allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(self.pardict[outgrid], "DerCloop_%s.npy" % gridname), allow_pickle=True,
                    )
                else:
                    linmod = np.load(
                        os.path.join(self.pardict[outgrid], "DerPlin_%s.npy" % gridname), allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(self.pardict[outgrid], "DerPloop_%s.npy" % gridname), allow_pickle=True,
                    )
            kin = linmod[0][0, :, 0]
            #print("kin from load_model_individual")
            #print(kin)
        else:
            if self.template:
                lintab, looptab = get_template_grids(self.pardict, pad=False, cf=self.pardict["do_corr"])
                paramsmod = None
                kin = lintab[..., 0, :, 0][(0,) * 4]
            else:
                paramstab, lintab, looptab = get_grids(self.pardict, pad=False, cf=self.pardict["do_corr"])
                paramsmod = sp.interpolate.RegularGridInterpolator(self.truecrd, paramstab)
                kin = lintab[..., 0, :, 0][(0,) * len(self.pardict["freepar"])]
            linmod = sp.interpolate.RegularGridInterpolator(self.truecrd, lintab)
            loopmod = sp.interpolate.RegularGridInterpolator(self.truecrd, looptab)

        return kin, paramsmod, linmod, loopmod

    def compute_params(self, coords):

    	if self.pardict["taylor_order"]:
            dtheta = np.array(coords) - self.valueref
            Params = get_ParamsTaylor(dtheta, self.paramsmod, self.pardict["taylor_order"])
    	else:
            Params = self.paramsmod(coords)[0]

    	return Params

    def compute_pk_separategrids(self, coords):

	#self.kin, self.paramsmod, self.linmod, self.loopmod = self.load_model()
        self.kin, self.paramsmod_NGC, self.linmod_NGC, self.loopmod_NGC, self.paramsmod_SGC, self.linmod_SGC, self.loopmod_SGC = self.load_model_NGC_SGC()

        if self.pardict["taylor_order"]:
            dtheta = np.array(coords) - self.valueref
            #Plin = get_PSTaylor(dtheta, self.linmod, self.pardict["taylor_order"])
            #Ploop = get_PSTaylor(dtheta, self.loopmod, self.pardict["taylor_order"])
            Plin_NGC = get_PSTaylor(dtheta, self.linmod_NGC, self.pardict["taylor_order"])
            Ploop_NGC = get_PSTaylor(dtheta, self.loopmod_NGC, self.pardict["taylor_order"])
            Plin_SGC = get_PSTaylor(dtheta, self.linmod_SGC, self.pardict["taylor_order"])
            Ploop_SGC = get_PSTaylor(dtheta, self.loopmod_SGC, self.pardict["taylor_order"])
        else:
            Plin = self.linmod(coords)[0]
            Ploop = self.loopmod(coords)[0]
        Plin_NGC = np.swapaxes(Plin_NGC, axis1=1, axis2=2)[:, 1:, :]
        Ploop_NGC = np.swapaxes(Ploop_NGC, axis1=1, axis2=2)[:, 1:, :]
        Plin_SGC = np.swapaxes(Plin_SGC, axis1=1, axis2=2)[:, 1:, :]
        Ploop_SGC = np.swapaxes(Ploop_SGC, axis1=1, axis2=2)[:, 1:, :]

        return Plin_NGC, Ploop_NGC, Plin_SGC, Ploop_SGC

    def compute_model_direct(self, coords):

        parameters = copy.deepcopy(self.pardict)

        for k, var in enumerate(self.pardict["freepar"]):
            parameters[var] = coords[k]
        if parameters["code"] == "CAMB":
            kin, Pin, Om, Da, Hz, fN, sigma8, sigma12, r_d = run_camb(parameters)
        else:
            kin, Pin, Om, Da, Hz, fN, sigma8, sigma12, r_d = run_class(parameters)

        # Get non-linear power spectrum from pybird
        self.correlator.compute(
            {"k11": kin, "P11": Pin, "z": float(self.pardict["z_pk"]), "Omega0_m": Om, "f": fN, "DA": Da, "H": Hz}
        )
        Plin, Ploop = (
            self.correlator.bird.formatTaylorCf() if self.pardict["do_corr"] else self.correlator.bird.formatTaylorPs()
        )

        Plin = np.swapaxes(Plin.reshape((3, Plin.shape[-2] // 3, Plin.shape[-1])), axis1=1, axis2=2)[:, 1:, :]
        Ploop = np.swapaxes(Ploop.reshape((3, Ploop.shape[-2] // 3, Ploop.shape[-1])), axis1=1, axis2=2)[:, 1:, :]

        return Plin, Ploop

    def compute_model_separategrids(self, cvals_NGC, cvals_SGC, plin_NGC, ploop_NGC, plin_SGC, ploop_SGC, x_data):
        P_model_NGC, P_model_interp_NGC = self.compute_model_individual(cvals_NGC, plin_NGC, ploop_NGC, x_data)
        P_model_SGC, P_model_interp_SGC = self.compute_model_individual(cvals_SGC, plin_SGC, ploop_SGC, x_data)
        P_model = np.concatenate((P_model_NGC, P_model_SGC))
        P_model_interp = np.concatenate((P_model_interp_NGC, P_model_interp_SGC))
        return P_model, P_model_interp, P_model_NGC, P_model_interp_NGC, P_model_SGC, P_model_interp_SGC

    def compute_model_individual(self, cvals, plin, ploop, x_data):

        plin0, plin2, plin4 = plin
        ploop0, ploop2, ploop4 = ploop

        b1, b2, b3, b4, cct, cr1, cr2, ce1, cemono, cequad, bnlo = cvals

        # the columns of the Ploop data files.
        cvals = np.array(
            [
                1,
                b1,
                b2,
                b3,
                b4,
                b1 * b1,
                b1 * b2,
                b1 * b3,
                b1 * b4,
                b2 * b2,
                b2 * b4,
                b4 * b4,
                2.0 * b1 * cct / self.k_nl ** 2,
                2.0 * b1 * cr1 / self.k_m ** 2,
                2.0 * b1 * cr2 / self.k_m ** 2,
                2.0 * cct / self.k_nl ** 2,
                2.0 * cr1 / self.k_m ** 2,
                2.0 * cr2 / self.k_m ** 2,
                2.0 * b1 ** 2 * bnlo / self.k_m ** 4,
            ]
        )

        P0 = np.dot(cvals, ploop0) + plin0[0] + b1 * plin0[1] + b1 * b1 * plin0[2]
        P2 = np.dot(cvals, ploop2) + plin2[0] + b1 * plin2[1] + b1 * b1 * plin2[2]
        if self.pardict["do_hex"]:
            P4 = np.dot(cvals, ploop4) + plin4[0] + b1 * plin4[1] + b1 * b1 * plin4[2]

        P0_interp = sp.interpolate.splev(x_data[0], sp.interpolate.splrep(self.kin, P0))
        P2_interp = sp.interpolate.splev(x_data[1], sp.interpolate.splrep(self.kin, P2))
        if self.pardict["do_hex"]:
            P4_interp = sp.interpolate.splev(x_data[2], sp.interpolate.splrep(self.kin, P4))

        if self.pardict["do_corr"]:
            C0 = np.exp(-self.k_m * x_data[0]) * self.k_m ** 2 / (4.0 * np.pi * x_data[0])
            C1 = -self.k_m ** 2 * np.exp(-self.k_m * x_data[0]) / (4.0 * np.pi * x_data[0] ** 2)
            C2 = (
                np.exp(-self.k_m * x_data[1])
                * (3.0 + 3.0 * self.k_m * x_data[1] + self.k_m ** 2 * x_data[1] ** 2)
                / (4.0 * np.pi * x_data[1] ** 3)
            )

            P0_interp += ce1 * C0 + cemono * C1
            P2_interp += cequad * C2
        else:
            P0_interp += ce1 + cemono * x_data[0] ** 2 / self.k_m ** 2
            P2_interp += cequad * x_data[1] ** 2 / self.k_m ** 2

        if self.pardict["do_hex"]:
            P_model = np.concatenate([P0, P2, P4])
            P_model_interp = np.concatenate([P0_interp, P2_interp, P4_interp])
        else:
            P_model = np.concatenate([P0, P2])
            P_model_interp = np.concatenate([P0_interp, P2_interp])

        return P_model, P_model_interp

    def compute_chi2(self, P_model, Pi, data):

        if self.pardict["do_marg"]:

            Covbi = np.dot(Pi, np.dot(data["cov_inv"], Pi.T))
            Covbi += self.priormat
            Cinvbi = np.linalg.inv(Covbi)
            vectorbi = np.dot(P_model, np.dot(data["cov_inv"], Pi.T)) - np.dot(data["invcovdata"], Pi.T)
            chi2nomar = (
                np.dot(P_model, np.dot(data["cov_inv"], P_model))
                - 2.0 * np.dot(data["invcovdata"], P_model)
                + data["chi2data"]
            )
            chi2mar = -np.dot(vectorbi, np.dot(Cinvbi, vectorbi)) + np.log(np.linalg.det(Covbi))
            chi_squared = chi2nomar + chi2mar

        else:

            # Compute the chi_squared
            chi_squared = 0.0
            for i in range(len(data["fit_data_combined"])):
                chi_squared += (P_model[i] - data["fit_data_combined"][i]) * np.sum(
                    data["cov_inv"][i, 0:] * (P_model - data["fit_data_combined"])
                )

        return chi_squared

    def get_Pi_for_marg_separategrids(self, ploop_NGC, ploop_SGC, b1_NGC, b1_SGC, shot_noise_NGC, shot_noise_SGC, x_data):
        Pi_NGC = self.get_Pi_for_marg_individual(ploop_NGC, b1_NGC, shot_noise_NGC, x_data)
        Pi_SGC = self.get_Pi_for_marg_individual(ploop_SGC, b1_SGC, shot_noise_SGC, x_data)
        #print(len(x_data[0]), len(x_data[1]))
        #print(ploop_NGC)
        zeros = np.zeros((8, (len(x_data[0]) + len(x_data[1]))))
        Pi_NGC_concatenated = np.concatenate((Pi_NGC, zeros), axis=0)
        Pi_SGC_concatenated = np.concatenate((zeros, Pi_SGC), axis=0)
        Pi = np.hstack((Pi_NGC_concatenated, Pi_SGC_concatenated))
        return Pi_NGC, Pi_SGC, Pi

    # Ignore names, works for both power spectrum and correlation function
    def get_Pi_for_marg_individual(self, ploop, b1, shot_noise, x_data):

        if self.pardict["do_marg"]:

            ploop0, ploop2, ploop4 = ploop

            Pb3 = np.concatenate(
                [
                    splev(x_data[0], splrep(self.kin, ploop0[3] + b1 * ploop0[7])),
                    splev(x_data[1], splrep(self.kin, ploop2[3] + b1 * ploop2[7])),
                ]
            )
            Pcct = np.concatenate(
                [
                    splev(x_data[0], splrep(self.kin, ploop0[15] + b1 * ploop0[12])),
                    splev(x_data[1], splrep(self.kin, ploop2[15] + b1 * ploop2[12])),
                ]
            )
            Pcr1 = np.concatenate(
                [
                    splev(x_data[0], splrep(self.kin, ploop0[16] + b1 * ploop0[13])),
                    splev(x_data[1], splrep(self.kin, ploop2[16] + b1 * ploop2[13])),
                ]
            )
            Pcr2 = np.concatenate(
                [
                    splev(x_data[0], splrep(self.kin, ploop0[17] + b1 * ploop0[14])),
                    splev(x_data[1], splrep(self.kin, ploop2[17] + b1 * ploop2[14])),
                ]
            )
            Pnlo = np.concatenate(
                [
                    splev(x_data[0], splrep(self.kin, b1 ** 2 * ploop0[18])),
                    splev(x_data[1], splrep(self.kin, b1 ** 2 * ploop2[18])),
                ]
            )

            if self.pardict["do_hex"]:

                Pb3 = np.concatenate([Pb3, splev(x_data[2], splrep(self.kin, ploop4[3] + b1 * ploop4[7]))])
                Pcct = np.concatenate([Pcct, splev(x_data[2], splrep(self.kin, ploop4[15] + b1 * ploop4[12]))])
                Pcr1 = np.concatenate([Pcr1, splev(x_data[2], splrep(self.kin, ploop4[16] + b1 * ploop4[13]))])
                Pcr2 = np.concatenate([Pcr2, splev(x_data[2], splrep(self.kin, ploop4[17] + b1 * ploop4[14]))])
                Pnlo = np.concatenate([Pnlo, splev(x_data[2], splrep(self.kin, b1 ** 2 * ploop4[18]))])

            if self.pardict["do_corr"]:

                C0 = np.concatenate(
                    [np.exp(-self.k_m * x_data[0]) / (4.0 * np.pi * x_data[0]), np.zeros(len(x_data[1]))]
                )  # shot-noise mono
                C1 = np.concatenate(
                    [-np.exp(-self.k_m * x_data[0]) / (4.0 * np.pi * x_data[0] ** 2), np.zeros(len(x_data[1]))]
                )  # k^2 mono
                C2 = np.concatenate(
                    [
                        np.zeros(len(x_data[0])),
                        np.exp(-self.k_m * x_data[1])
                        * (3.0 + 3.0 * self.k_m * x_data[1] + self.k_m ** 2 * x_data[1] ** 2)
                        / (4.0 * np.pi * x_data[1] ** 3),
                    ]
                )  # k^2 quad

                if self.pardict["do_hex"]:
                    C0 = np.concatenate([C0, np.zeros(len(x_data[2]))])  # shot-noise mono
                    C1 = np.concatenate([C1, np.zeros(len(x_data[2]))])  # k^2 mono
                    C2 = np.concatenate([C2, np.zeros(len(x_data[2]))])  # k^2 quad

                Pi = np.array(
                    [
                        Pb3,  # *b3
                        2.0 * Pcct / self.k_nl ** 2,  # *cct
                        2.0 * Pcr1 / self.k_m ** 2,  # *cr1
                        2.0 * Pcr2 / self.k_m ** 2,  # *cr2
                        C0 * self.k_m ** 2 * shot_noise,  # ce1
                        C1 * self.k_m ** 2 * shot_noise,  # cemono
                        C2 * shot_noise,  # cequad
                        2.0 * Pnlo / self.k_m ** 4,  # bnlo
                    ]
                )

            else:

                Onel0 = np.concatenate([np.ones(len(x_data[0])), np.zeros(len(x_data[1]))])  # shot-noise mono
                kl0 = np.concatenate([x_data[0], np.zeros(len(x_data[1]))])  # k^2 mono
                kl2 = np.concatenate([np.zeros(len(x_data[0])), x_data[1]])  # k^2 quad

                if self.pardict["do_hex"]:
                    Onel0 = np.concatenate([Onel0, np.zeros(len(x_data[2]))])  # shot-noise mono
                    kl0 = np.concatenate([kl0, np.zeros(len(x_data[2]))])  # k^2 mono
                    kl2 = np.concatenate([kl2, np.zeros(len(x_data[2]))])  # k^2 quad

                Pi = np.array(
                    [
                        Pb3,  # *b3
                        2.0 * Pcct / self.k_nl ** 2,  # *cct
                        2.0 * Pcr1 / self.k_m ** 2,  # *cr1
                        2.0 * Pcr2 / self.k_m ** 2,  # *cr2
                        Onel0 * shot_noise,  # *ce1
                        kl0 ** 2 / self.k_m ** 2 * shot_noise,  # *cemono
                        kl2 ** 2 / self.k_m ** 2 * shot_noise,  # *cequad
                        2.0 * Pnlo / self.k_m ** 4,  # bnlo
                    ]
                )

        else:

            Pi = None

        return Pi

    def compute_bestfit_analytic(self, Pi, data, model):

        Covbi = np.dot(Pi, np.dot(data["cov_inv"], Pi.T))
        Covbi += self.priormat
        Cinvbi = np.linalg.inv(Covbi)
        vectorbi = Pi @ data["cov_inv"] @ (data["fit_data_combined"] - model)

        return Cinvbi @ vectorbi

    def get_components(self, coords, cvals):

        if self.direct:
            plin, ploop = self.compute_model_direct(coords)
        else:
            plin, ploop = self.compute_pk(coords)

        plin0, plin2, plin4 = plin
        ploop0, ploop2, ploop4 = ploop

        b1, b2, b3, b4, cct, cr1, cr2, ce1, cemono, cequad, bnlo = cvals

        # the columns of the Ploop data files.
        cloop = np.array([1, b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3, b1 * b4, b2 * b2, b2 * b4, b4 * b4])
        cvalsct = np.array(
            [
                2.0 * b1 * cct / self.k_nl ** 2,
                2.0 * b1 * cr1 / self.k_m ** 2,
                2.0 * b1 * cr2 / self.k_m ** 2,
                2.0 * cct / self.k_nl ** 2,
                2.0 * cr1 / self.k_m ** 2,
                2.0 * cr2 / self.k_m ** 2,
            ]
        )
        cnlo = 2.0 * b1 ** 2 * bnlo / self.k_m ** 4

        P0lin = plin0[0] + b1 * plin0[1] + b1 * b1 * plin0[2]
        P2lin = plin2[0] + b1 * plin2[1] + b1 * b1 * plin2[2]
        P0loop = np.dot(cloop, ploop0[:12, :])
        P2loop = np.dot(cloop, ploop2[:12, :])
        P0ct = np.dot(cvalsct, ploop0[12:-1, :])
        P2ct = np.dot(cvalsct, ploop2[12:-1, :])
        P0nlo = cnlo * ploop0[-1, :]
        P2nlo = cnlo * ploop2[-1, :]
        if self.pardict["do_hex"]:
            P4lin = plin4[0] + b1 * plin4[1] + b1 * b1 * plin4[2]
            P4loop = np.dot(cloop, ploop4[:12, :])
            P4ct = np.dot(cvalsct, ploop4[12:-1, :])
            P4nlo = cnlo * ploop4[-1, :]
            Plin = [P0lin, P2lin, P4lin]
            Ploop = [P0loop + P0nlo, P2loop + P2nlo, P4loop + P4nlo]
            Pct = [P0ct, P2ct, P4ct]
        else:
            Plin = [P0lin, P2lin]
            Ploop = [P0loop + P0nlo, P2loop + P2nlo]
            Pct = [P0ct, P2ct]

        if self.pardict["do_corr"]:
            C0 = np.exp(-self.k_m * self.kin) * self.k_m ** 2 / (4.0 * np.pi * self.kin)
            C1 = -self.k_m ** 2 * np.exp(-self.k_m * self.kin) / (4.0 * np.pi * self.kin ** 2)
            C2 = (
                np.exp(-self.k_m * self.kin)
                * (3.0 + 3.0 * self.k_m * self.kin + self.k_m ** 2 * self.kin ** 2)
                / (4.0 * np.pi * self.kin ** 3)
            )
            P0st = ce1 * C0 + cemono * C1
            P2st = cequad * C2
            P4st = np.zeros(len(self.kin))
        else:
            P0st = ce1 + cemono * self.kin ** 2 / self.k_m ** 2
            P2st = cequad * self.kin ** 2 / self.k_m ** 2
            P4st = np.zeros(len(self.kin))
        if self.pardict["do_hex"]:
            Pst = [P0st, P2st, P4st]
        else:
            Pst = [P0st, P2st]

        return Plin, Ploop, Pct, Pst


# Holds all the data in a convenient dictionary
class FittingData_NGC_SGC:
    def __init__(self, pardict_NGC_SGC, NGC_shot_noise, SGC_shot_noise):

        x_data, fit_data_combined, fit_data_NGC, fit_data_SGC, cov, cov_inv, chi2data, invcovdata, fitmask_NGC, fitmask_SGC = self.read_data(pardict_NGC_SGC)

        self.data = {
            "x_data": x_data,
            "fit_data_combined": fit_data_combined,
            "fit_data_NGC": fit_data_NGC,
            "fit_data_SGC": fit_data_SGC,
            "cov": cov,
            "cov_inv": cov_inv,
            "chi2data": chi2data,
            "invcovdata": invcovdata,
            "fitmask_NGC": fitmask_NGC,
            "fitmask_SGC": fitmask_SGC,
            "shot_noise_NGC": NGC_shot_noise,
            "shot_noise_SGC": SGC_shot_noise,
        }

        # Check covariance matrix is symmetric and positive-definite by trying to do a cholesky decomposition
        diff = np.abs((self.data["cov"] - self.data["cov"].T) / self.data["cov"])
        if not (np.logical_or(diff <= 1.0e-6, np.isnan(diff))).all():
            print(diff)
            print("Error: Covariance matrix not symmetric!")
            exit(0)
        try:
            cholesky(self.data["cov"])
        except:
            print("Error: Covariance matrix not positive-definite!")
            exit(0)

    def read_pk_single(self, inputfile, step_size, skiprows):

        dataframe = pd.read_csv(
            inputfile,
            comment="#",
            skiprows=skiprows,
            delim_whitespace=True,
            names=["k", "k_mean", "pk0", "pk2", "pk4", "nk"], #format of my power spectra
            #names = ["n","k","k_mean","pk0","pk2","pk4","sigma_lin_pk","nk"],#format of mock mean
        )
        k = dataframe["k"].values
        if step_size == 1:
            k_rebinned = k
            pk0_rebinned = dataframe["pk0"].values
            pk2_rebinned = dataframe["pk2"].values
            pk4_rebinned = dataframe["pk4"].values
        else:
            add = k.size % step_size
            weight = dataframe["nk"].values
            if add:
                to_add = step_size - add
                k = np.concatenate((k, [k[-1]] * to_add))
                dataframe["pk0"].values = np.concatenate(
                    (dataframe["pk0"].values, [dataframe["pk0"].values[-1]] * to_add)
                )
                dataframe["pk2"].values = np.concatenate(
                    (dataframe["pk2"].values, [dataframe["pk2"].values[-1]] * to_add)
                )
                dataframe["pk4"].values = np.concatenate(
                    (dataframe["pk4"].values, [dataframe["pk4"].values[-1]] * to_add)
                )
                weight = np.concatenate((weight, [0] * to_add))
            k = k.reshape((-1, step_size))
            pk0 = (dataframe["pk0"].values).reshape((-1, step_size))
            pk2 = (dataframe["pk2"].values).reshape((-1, step_size))
            pk4 = (dataframe["pk4"].values).reshape((-1, step_size))
            weight = weight.reshape((-1, step_size))
            # Take the average of every group of step_size rows to rebin
            k_rebinned = np.average(k, axis=1)
            pk0_rebinned = np.average(pk0, axis=1, weights=weight)
            pk2_rebinned = np.average(pk2, axis=1, weights=weight)
            pk4_rebinned = np.average(pk4, axis=1, weights=weight)

        #return np.vstack([k_rebinned, pk0_rebinned, pk2_rebinned, pk4_rebinned]).T
        return np.vstack([k_rebinned, pk0_rebinned, pk2_rebinned]).T

    def read_pk(self, inputfile_NGC, inputfile_SGC, step_size, skiprows):
        read_pk_NGC = self.read_pk_single(inputfile_NGC, step_size, skiprows)
        read_pk_SGC = self.read_pk_single(inputfile_SGC, step_size, skiprows)
        #print("NGC =")
        #print(read_pk_NGC)
        #print("SGC =")
        #print(read_pk_SGC)
        pk_NGC_SGC = np.vstack((read_pk_NGC[:, 0], read_pk_NGC[:, 1], read_pk_NGC[:, 2], read_pk_SGC[:, 1], read_pk_SGC[:, 2])).T
        #print("pk_NGC_SGC")
        #print(pk_NGC_SGC)
        return read_pk_NGC, read_pk_SGC, pk_NGC_SGC

    def read_data(self, pardict_NGC_SGC):

        # Read in the data
        #print(pardict_NGC["datafile"])
        #print(pardict_SGC["datafile"])
        #if pardict_NGC["do_corr"]:
        #    data = np.array(pd.read_csv(pardict_NGC["datafile"], delim_whitespace=True, header=None))
        #else:
            #data = self.read_pk(pardict["datafile"], 1, 10) #Original- skips the first 10 lines of DESI stage 2 data
        data_NGC, data_SGC, combined_data = self.read_pk(pardict_NGC_SGC["datafile_NGC"], pardict_NGC_SGC["datafile_SGC"], 1, 0)
        
            #print(data)

        x_data_NGC = data_NGC[:, 0]
        x_data_SGC = data_SGC[:, 0]
        fitmask_NGC = [
            (np.where(np.logical_and(x_data_NGC >= pardict_NGC_SGC["xfit_min"][0], x_data_NGC <= pardict_NGC_SGC["xfit_max"][0]))[0]).astype(
                int
            ),
            (np.where(np.logical_and(x_data_NGC >= pardict_NGC_SGC["xfit_min"][1], x_data_NGC <= pardict_NGC_SGC["xfit_max"][1]))[0]).astype(
                int
            ),
            (np.where(np.logical_and(x_data_NGC >= pardict_NGC_SGC["xfit_min"][2], x_data_NGC <= pardict_NGC_SGC["xfit_max"][2]))[0]).astype(
                int
            ),
        ]
        
        fitmask_SGC = [
            (np.where(np.logical_and(x_data_SGC >= pardict_NGC_SGC["xfit_min"][0], x_data_SGC <= pardict_NGC_SGC["xfit_max"][0]))[0]).astype(
                int
            ),
            (np.where(np.logical_and(x_data_SGC >= pardict_NGC_SGC["xfit_min"][1], x_data_SGC <= pardict_NGC_SGC["xfit_max"][1]))[0]).astype(
                int
            ),
            (np.where(np.logical_and(x_data_SGC >= pardict_NGC_SGC["xfit_min"][2], x_data_SGC <= pardict_NGC_SGC["xfit_max"][2]))[0]).astype(
                int
            ),
        ]
        
        x_data_NGC = [data_NGC[fitmask_NGC[0], 0], data_NGC[fitmask_NGC[1], 0], data_NGC[fitmask_NGC[2], 0]]
        x_data_SGC = [data_SGC[fitmask_SGC[0], 0], data_SGC[fitmask_SGC[1], 0], data_SGC[fitmask_SGC[2], 0]]
        if pardict_NGC_SGC["do_hex"]:
            fit_data = np.concatenate([data_NGC[fitmask_NGC[0], 1], data_NGC[fitmask_NGC[1], 2], data_NGC[fitmask_NGC[2], 3]])
        else:
            fit_data_NGC = np.concatenate([data_NGC[fitmask_NGC[0], 1], data_NGC[fitmask_NGC[1], 2]])
            fit_data_SGC = np.concatenate([data_SGC[fitmask_SGC[0], 1], data_SGC[fitmask_SGC[1], 2]])
            fit_data = np.concatenate([data_NGC[fitmask_NGC[0], 1], data_NGC[fitmask_NGC[1], 2], data_SGC[fitmask_SGC[0], 1], data_SGC[fitmask_SGC[1], 2]])
            
            
        #print("x_data (combined) = ")
        #print(x_data_NGC)
        #print("fit_data (combined) = ")
        #print(fit_data)

        # Read in, reshape and mask the covariance matrix
        cov_flat = np.array(pd.read_csv(pardict_NGC_SGC["covfile"], delim_whitespace=True, header=None)) #ORIGINAL COV MATRIX FLAT
        cov_flat = cov_flat.flatten() #SO I TRY FLATTENING MINE
        nin = len(data_NGC[:, 0])
        #print("nin = %lf" %nin)
        #cov_input = cov_flat[:, 2].reshape((3 * nin, 3 * nin)) #ORIGINAL HAD TWO COLUMNS, MY FLATTENED GRID DOES NOT
        #cov_input = cov_flat.reshape((3 * nin, 3 * nin)) #Only for mono+quad+hex
        #cov_input = cov_flat.reshape((2 * nin, 2 * nin)) #Only for mono+quad
        cov_input = cov_flat.reshape((4 * nin, 4 * nin)) #Only for NGC mono, NGC quad, SGC Mono, SGC Quad
        nx0, nx2 = len(x_data_NGC[0]), len(x_data_NGC[1])
        #print("nx0 = %lf" %nx0)
        nx4 = len(x_data_NGC[2]) if pardict_NGC_SGC["do_hex"] else 0
        mask0_NGC, mask2_NGC, mask4_NGC = fitmask_NGC[0][:, None], fitmask_NGC[1][:, None], fitmask_NGC[2][:, None]
        mask0_SGC, mask2_SGC, mask4_SGC = fitmask_SGC[0][:, None], fitmask_SGC[1][:, None], fitmask_SGC[2][:, None]
        #cov = np.zeros((nx0 + nx2 + nx4, nx0 + nx2 + nx4)) #original, update for P0, P2, P0, P2 config
        cov = np.zeros((nx0 + nx2 + nx0 + nx2, nx0 + nx2 + nx0 + nx2))
        
        # 1  | 2  | 3  | 4
        # 5  | 6  | 7  | 8
        # 9  | 10 | 11 | 12
        # 13 | 14 | 15 | 16
        
        cov[:nx0, :nx0] = cov_input[mask0_NGC, mask0_NGC.T] # 1
        cov[:nx0, nx0 : nx0 + nx2] = cov_input[mask0_NGC, nin + mask2_NGC.T] #2
        cov[:nx0, nx0+nx2 : nx0+nx2+nx0] = cov_input[mask0_NGC, 2*nin + mask0_SGC.T] #3
        cov[:nx0, nx0+nx2+nx0 : nx0+nx2+nx0+nx2] = cov_input[mask0_NGC, 3*nin + mask2_SGC.T] #4
        
        cov[nx0:nx0+nx2, :nx0] = cov_input[nin + mask2_NGC, mask0_NGC.T] #5
        cov[nx0:nx0+nx2, nx0:nx0+nx2] = cov_input[nin + mask2_NGC, nin+mask2_NGC.T] #6
        cov[nx0:nx0+nx2, nx0+nx2:nx0+nx2+nx0] = cov_input[nin + mask2_NGC, 2*nin+mask0_SGC.T] #7
        cov[nx0:nx0+nx2, nx0+nx2+nx0:nx0+nx2+nx0+nx2] = cov_input[nin + mask2_NGC, 3*nin + mask2_SGC.T] #8
        
        cov[nx0+nx2:nx0+nx2+nx0, :nx0] = cov_input[2*nin + mask0_SGC, mask0_NGC.T] #9
        cov[nx0+nx2:nx0+nx2+nx0, nx0:nx0+nx2] = cov_input[2*nin + mask0_SGC, nin + mask2_NGC.T] #10
        cov[nx0+nx2:nx0+nx2+nx0, nx0+nx2:nx0+nx2+nx0] = cov_input[2*nin + mask0_SGC, 2*nin+mask0_SGC.T] #11
        cov[nx0+nx2:nx0+nx2+nx0, nx0+nx2+nx0:nx0+nx2+nx0+nx2]= cov_input[2*nin + mask0_SGC, 3*nin+mask2_SGC.T] #12
        
        cov[nx0+nx2+nx0:nx0+nx2+nx0+nx2, :nx0] = cov_input[3*nin + mask2_SGC, mask0_NGC.T] #13
        cov[nx0+nx2+nx0:nx0+nx2+nx0+nx2, nx0:nx0+nx2] = cov_input[3*nin+mask2_SGC, nin+mask2_NGC.T] #14
        cov[nx0+nx2+nx0:nx0+nx2+nx0+nx2, nx0+nx2:nx0+nx2+nx0] = cov_input[3*nin+mask2_SGC, 2*nin+mask0_SGC.T] #15
        cov[nx0+nx2+nx0:nx0+nx2+nx0+nx2, nx0+nx2+nx0:nx0+nx2+nx0+nx2] = cov_input[3*nin+mask2_SGC, 3*nin+mask2_SGC.T] #16
        
        #if pardict["do_hex"]:
        #    cov[:nx0, nx0 + nx2 :] = cov_input[mask0, 2 * nin + mask4.T]
        #    cov[nx0 + nx2 :, :nx0] = cov_input[2 * nin + mask4, mask0.T]
        #    cov[nx0 : nx0 + nx2, nx0 + nx2 :] = cov_input[nin + mask2, 2 * nin + mask4.T]
        #    cov[nx0 + nx2 :, nx0 : nx0 + nx2] = cov_input[2 * nin + mask4, nin + mask2.T]
        #    cov[nx0 + nx2 :, nx0 + nx2 :] = cov_input[2 * nin + mask4, 2 * nin + mask4.T]

        # Invert the covariance matrix
        #identity = np.eye(nx0 + nx2 + nx4) #testing changing this for mono+quad only?
        identity = np.eye(nx0 + nx2 + nx0 + nx2) #for P0, P2, P0, P2
        #print("Cov:")
        #print(cov)
        #print("Identity:")
        #print(nx2)
        cov_lu, pivots, cov_inv, info = lapack.dgesv(cov, identity)
        #print("cov_inv = ")
        #print(cov_inv)
        #print("length of cov_inv = %lf" %len(cov_inv[0]))

        chi2data = np.dot(fit_data, np.dot(cov_inv, fit_data))
        invcovdata = np.dot(fit_data, cov_inv)
        np.savetxt("cov_input.txt", cov_input)
        np.savetxt("cov_mask.txt", cov)
        #print("chi2data = %lf" %chi2data)

        return x_data_NGC, fit_data, fit_data_NGC, fit_data_SGC, cov, cov_inv, chi2data, invcovdata, fitmask_NGC, fitmask_SGC
    def read_data_original(self, pardict):

        """# Read in the first mock to allocate the arrays
        skiprows = 0
        nmocks = 1000
        inputbase = "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/input_data/EZmock_xil_v2"
        inputfile = str("%s/2PCF_20200514-unit-elg-3gpc-001.dat" % inputbase)
        data = np.array(pd.read_csv(inputfile, delim_whitespace=True, dtype=float, header=None, skiprows=skiprows))
        sdata = data[:, 0]

        xi = np.empty((nmocks, 4 * len(sdata)))
        for i in range(nmocks):
            inputfile = str("%s/2PCF_20200514-unit-elg-3gpc-%.3d.dat" % (inputbase, i))
            data = np.array(pd.read_csv(inputfile, delim_whitespace=True, dtype=float, header=None, skiprows=skiprows))
            xi[i] = np.concatenate([data[:, 0], data[:, 1], data[:, 2], data[:, 3]])

        data = np.mean(xi, axis=0)
        data = data.reshape((4, len(sdata))).T
        cov_input = np.cov(xi[:, len(data[:, 0]) :].T)
        print(cov_input)"""

        # Read in the data
        #print(pardict["datafile"])
        if pardict["do_corr"]:
            data = np.array(pd.read_csv(pardict["datafile"], delim_whitespace=True, header=None))
        else:
            #data = self.read_pk(pardict["datafile"], 1, 10)
            data = self.read_pk(pardict["datafile"], 1, 0) #no need to skip lines

        x_data = data[:, 0]
        fitmask = [
            (np.where(np.logical_and(x_data >= pardict["xfit_min"][0], x_data <= pardict["xfit_max"][0]))[0]).astype(
                int
            ),
            (np.where(np.logical_and(x_data >= pardict["xfit_min"][1], x_data <= pardict["xfit_max"][1]))[0]).astype(
                int
            ),
            (np.where(np.logical_and(x_data >= pardict["xfit_min"][2], x_data <= pardict["xfit_max"][2]))[0]).astype(
                int
            ),
        ]
        x_data = [data[fitmask[0], 0], data[fitmask[1], 0], data[fitmask[2], 0]]
        if pardict["do_hex"]:
            fit_data = np.concatenate([data[fitmask[0], 1], data[fitmask[1], 2], data[fitmask[2], 3]])
        else:
            fit_data = np.concatenate([data[fitmask[0], 1], data[fitmask[1], 2]])

        # Read in, reshape and mask the covariance matrix
        #cov_flat = np.array(pd.read_csv(pardict["covfile"], delim_whitespace=True, header=None))
        #nin = len(data[:, 0])
        #cov_input = cov_flat[:, 2].reshape((3 * nin, 3 * nin))

        cov_input = np.array(pd.read_csv(pardict["covfile"], delim_whitespace=True, header=None)) #+
        np.savetxt("cov_input_imshow.txt", cov_input)
        #cov_flat = cov_flat.flatten() #+
        nin = len(data[:, 0]) #+
        print(nin)

        nx0, nx2 = len(x_data[0]), len(x_data[1])
        nx4 = len(x_data[2]) if pardict["do_hex"] else 0
        mask0, mask2, mask4 = fitmask[0][:, None], fitmask[1][:, None], fitmask[2][:, None]
        cov = np.zeros((nx0 + nx2 + nx4, nx0 + nx2 + nx4))
        cov[:nx0, :nx0] = cov_input[mask0, mask0.T]
        cov[:nx0, nx0 : nx0 + nx2] = cov_input[mask0, nin + mask2.T]
        cov[nx0 : nx0 + nx2, :nx0] = cov_input[nin + mask2, mask0.T]
        cov[nx0 : nx0 + nx2, nx0 : nx0 + nx2] = cov_input[nin + mask2, nin + mask2.T]
        if pardict["do_hex"]:
            cov[:nx0, nx0 + nx2 :] = cov_input[mask0, 2 * nin + mask4.T]
            cov[nx0 + nx2 :, :nx0] = cov_input[2 * nin + mask4, mask0.T]
            cov[nx0 : nx0 + nx2, nx0 + nx2 :] = cov_input[nin + mask2, 2 * nin + mask4.T]
            cov[nx0 + nx2 :, nx0 : nx0 + nx2] = cov_input[2 * nin + mask4, nin + mask2.T]
            cov[nx0 + nx2 :, nx0 + nx2 :] = cov_input[2 * nin + mask4, 2 * nin + mask4.T]

        # Invert the covariance matrix
        identity = np.eye(nx0 + nx2 + nx4)
        cov_lu, pivots, cov_inv, info = lapack.dgesv(cov, identity)

        chi2data = np.dot(fit_data, np.dot(cov_inv, fit_data))
        invcovdata = np.dot(fit_data, cov_inv)

        return x_data, fit_data, cov, cov_inv, chi2data, invcovdata, fitmask

def create_plot_NGC_SGC_z1_z3(pardict_NGC_z1, pardict_SGC_z1, pardict_NGC_z3, pardict_SGC_z3, colours, fittingdata_z1, fittingdata_z3):

    if pardict_NGC_z1["do_hex"]:
        x_data_z1 = fittingdata_z1.data["x_data_NGC"] #Only want to use 1 range to define nx0, nx2
        nx0_z1, nx2_z1, nx4_z1 = len(x_data_z1[0]), len(x_data_z1[1]), len(x_data_z1[2])
        #print("nx0 = %lf" %nx0_z1)
    else:
        #print(fittingdata_z1.data["x_data"][:2])
        x_data_z1 = fittingdata_z1.data["x_data"][:2]
        nx0_z1, nx2_z1, nx4_z1 = len(x_data_z1[0]), len(x_data_z1[1]), 0
    fit_data_NGC_z1 = fittingdata_z1.data["fit_data_NGC"]
    fit_data_SGC_z1 = fittingdata_z1.data["fit_data_SGC"]
    cov_z1 = fittingdata_z1.data["cov"]

    if pardict_NGC_z3["do_hex"]:
        x_data_z3 = fittingdata_z3.data["x_data_NGC"] #Only want to use 1 range to define nx0, nx2
        nx0_z3, nx2_z3, nx4_z3 = len(x_data_z3[0]), len(x_data_z3[1]), len(x_data_z3[2])
        #print("nx0 = %lf" %nx0_z3)
    else:
        #print(fittingdata_z3.data["x_data"][:2])
        x_data_z3 = fittingdata_z3.data["x_data"][:2]
        nx0_z3, nx2_z3, nx4_z3 = len(x_data_z1[0]), len(x_data_z3[1]), 0
    fit_data_NGC_z3 = fittingdata_z3.data["fit_data_NGC"]
    fit_data_SGC_z3 = fittingdata_z3.data["fit_data_SGC"]
    cov_z3 = fittingdata_z3.data["cov"]

    plt_data_NGC_z1 = (
        np.concatenate(x_data_z1) ** 2 * fit_data_NGC_z1 if pardict_NGC_z1["do_corr"] else np.concatenate(x_data_z1) * fit_data_NGC_z1 
    )
    plt_data_SGC_z1 = (
        np.concatenate(x_data_z1) ** 2 * fit_data_SGC_z1 if pardict_SGC_z1["do_corr"] else np.concatenate(x_data_z1) * fit_data_SGC_z1 
    )
    if pardict_NGC_z1["do_corr"]:
        plt_err_z1 = np.concatenate(x_data_z1) ** 2 * np.sqrt(cov_z1[np.diag_indices(nx0_z1 + nx2_z1 + nx4_z1)])
    else:
        plt_err_z1 = np.concatenate((x_data_z1, x_data_z1)).flatten() * np.sqrt(cov_z1[np.diag_indices(nx0_z1 + nx2_z1 + nx0_z1 + nx2_z1)])


    plt_data_NGC_z3 = (
        np.concatenate(x_data_z3) ** 2 * fit_data_NGC_z3 if pardict_NGC_z3["do_corr"] else np.concatenate(x_data_z3) * fit_data_NGC_z3 
    )
    plt_data_SGC_z3 = (
        np.concatenate(x_data_z3) ** 2 * fit_data_SGC_z3 if pardict_SGC_z3["do_corr"] else np.concatenate(x_data_z3) * fit_data_SGC_z3 
    )
    if pardict_NGC_z3["do_corr"]:
        plt_err_z3 = np.concatenate(x_data_z3) ** 2 * np.sqrt(cov_z3[np.diag_indices(nx0_z3 + nx2_z3 + nx4_z3)])
    else:
        plt_err_z3 = np.concatenate((x_data_z3, x_data_z3)).flatten() * np.sqrt(cov_z3[np.diag_indices(nx0_z3 + nx2_z3 + nx0_z3 + nx2_z3)])

    params = {'text.usetex': True}
    plt.rcParams.update(params)

#NGC Monopole + Quadrupole z1
    plt.errorbar(
        x_data_z1[0],
        plt_data_NGC_z1[:nx0_z1],
        yerr=plt_err_z1[:nx0_z1],
        marker="o",
        markerfacecolor=colours[0],
        markeredgecolor=colours[0],
        color=colours[0],
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )
    plt.errorbar(
        x_data_z1[1],
        plt_data_NGC_z1[nx0_z1 : nx0_z1 + nx2_z1],
        yerr=plt_err_z1[nx0_z1 : nx0_z1 + nx2_z1],
        marker="o",
        markerfacecolor=colours[0],
        markeredgecolor=colours[0],
        color=colours[0],
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )
    
#SGC Monopole + Quadrupole z1 (using distinct NGC/SGC errors from combined cov_matrix) 
    plt.errorbar(
        x_data_z1[0],
        plt_data_SGC_z1[:nx0_z1],
        yerr=plt_err_z1[nx0_z1 + nx2_z1 : nx0_z1 + nx2_z1 + nx0_z1],
        marker="o",
        markerfacecolor=colours[1],
        markeredgecolor=colours[1],
        color=colours[1],
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )
    plt.errorbar(
        x_data_z1[1],
        plt_data_SGC_z1[nx0_z1 : nx0_z1 + nx2_z1],
        yerr=plt_err_z1[nx0_z1 + nx0_z1 + nx2_z1 : nx0_z1 + nx2_z1 + nx0_z1 + nx2_z1],
        marker="o",
        markerfacecolor=colours[1],
        markeredgecolor=colours[1],
        color=colours[1],
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )    

    if pardict_NGC_z1["do_hex"]:
        plt.errorbar(
            x_data_z1[2],
            plt_data_NGC[nx0_z1 + nx2_z1 :],
            yerr=plt_err[nx0_z1 + nx2_z1 :],
            marker="o",
            markerfacecolor="g",
            markeredgecolor="k",
            color="g",
            linestyle="None",
            markeredgewidth=1.3,
            zorder=5,
        )

#--------------------------------------------------------------------------------------

#NGC Monopole + Quadrupole z3
    plt.errorbar(
        x_data_z3[0],
        plt_data_NGC_z3[:nx0_z3],
        yerr=plt_err_z3[:nx0_z3],
        marker="o",
        markerfacecolor=colours[2],
        markeredgecolor=colours[2],
        color=colours[2],
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )
    plt.errorbar(
        x_data_z3[1],
        plt_data_NGC_z3[nx0_z3 : nx0_z3 + nx2_z3],
        yerr=plt_err_z3[nx0_z3 : nx0_z3 + nx2_z3],
        marker="o",
        markerfacecolor=colours[2],
        markeredgecolor=colours[2],
        color=colours[2],
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )
    
#SGC Monopole + Quadrupole z1 (using distinct NGC/SGC errors from combined cov_matrix) 
    plt.errorbar(
        x_data_z3[0],
        plt_data_SGC_z3[:nx0_z3],
        yerr=plt_err_z3[nx0_z3 + nx2_z3 : nx0_z3 + nx2_z3 + nx0_z3],
        marker="o",
        markerfacecolor=colours[3],
        markeredgecolor=colours[3],
        color=colours[3],
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )
    plt.errorbar(
        x_data_z3[1],
        plt_data_SGC_z3[nx0_z3 : nx0_z3 + nx2_z3],
        yerr=plt_err_z3[nx0_z3 + nx0_z3 + nx2_z3 : nx0_z3 + nx2_z3 + nx0_z3 + nx2_z3],
        marker="o",
        markerfacecolor=colours[3],
        markeredgecolor=colours[3],
        color=colours[3],
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )    

    if pardict_NGC_z3["do_hex"]:
        plt.errorbar(
            x_data_z3[2],
            plt_data_NGC[nx0_z3 + nx2_z3 :],
            yerr=plt_err[nx0_z3 + nx2_z3 :],
            marker="o",
            markerfacecolor="g",
            markeredgecolor="k",
            color="g",
            linestyle="None",
            markeredgewidth=1.3,
            zorder=5,
        )

    plt.xlim(0.03, np.amax(pardict_NGC_z1["xfit_max"]) * 1.05)
    plt.ylim(100, 2000)
    if pardict_NGC_z1["do_corr"]:
        plt.xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$", fontsize=16)
        plt.ylabel(r"$s^{2}\xi(s)$", fontsize=16, labelpad=5)
    else:
        plt.xlabel(r"$k\,(h\,\mathrm{Mpc}^{-1})$", fontsize=16)
        plt.ylabel(r"$kP(k)\,(h^{-1}\,\mathrm{Mpc})$", fontsize=16, labelpad=5)
    plt.tick_params(width=1.3)
    plt.tick_params("both", length=10, which="major")
    plt.tick_params("both", length=5, which="minor")
    for axis in ["top", "left", "bottom", "right"]:
        plt.gca().spines[axis].set_linewidth(1.3)
    for tick in plt.gca().xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in plt.gca().yaxis.get_ticklabels():
        tick.set_fontsize(14)
    plt.tight_layout()
    plt.gca().set_autoscale_on(False)
    plt.ion()

    return plt

def create_plot_combined(pardict_NGC_SGC, colours, fittingdata):

    if pardict_NGC_SGC["do_hex"]:
        x_data = fittingdata.data["x_data_NGC"] #Only want to use 1 range to define nx0, nx2
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), len(x_data[2])
        print("nx0 = %lf" %nx0)
    else:
        print(fittingdata.data["x_data"][:2])
        x_data = fittingdata.data["x_data"][:2]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), 0
    fit_data_NGC = fittingdata.data["fit_data_NGC"]
    fit_data_SGC = fittingdata.data["fit_data_SGC"]
    cov = fittingdata.data["cov"]

    plt_data_NGC = (
        np.concatenate(x_data) ** 2 * fit_data_NGC if pardict_NGC_SGC["do_corr"] else np.concatenate(x_data) * fit_data_NGC 
    )
    plt_data_SGC = (
        np.concatenate(x_data) ** 2 * fit_data_SGC if pardict_NGC_SGC["do_corr"] else np.concatenate(x_data) * fit_data_SGC 
    )
    if pardict_NGC_SGC["do_corr"]:
        plt_err = np.concatenate(x_data) ** 2 * np.sqrt(cov[np.diag_indices(nx0 + nx2 + nx4)])
    else:
        plt_err = np.concatenate((x_data, x_data)).flatten() * np.sqrt(cov[np.diag_indices(nx0 + nx2 + nx0 + nx2)])

    params = {'text.usetex': True}
    plt.rcParams.update(params)

#NGC Monopole + Quadrupole
    plt.errorbar(
        x_data[0],
        plt_data_NGC[:nx0],
        yerr=plt_err[:nx0],
        marker="o",
        markerfacecolor=colours[0],
        markeredgecolor="k",
        color=colours[0],
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
        label=r"$\rm{NGC} \ P_{0}$"
    )
    plt.errorbar(
        x_data[1],
        plt_data_NGC[nx0 : nx0 + nx2],
        yerr=plt_err[nx0 : nx0 + nx2],
        marker="s",
        markerfacecolor=colours[0],
        markeredgecolor="k",
        color=colours[0],
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
        label=r"$\rm{NGC} \ P_{2}$"
    )
    
#SGC Monopole + Quadrupole (using distinct NGC/SGC errors from combined cov_matrix) 
    plt.errorbar(
        x_data[0],
        plt_data_SGC[:nx0],
        yerr=plt_err[nx0 + nx2 : nx0 + nx2 + nx0],
        marker="o",
        markerfacecolor=colours[1],
        markeredgecolor="k",
        color=colours[1],
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
        label=r"$\rm{SGC} \ P_{0}$"
    ) 
    plt.errorbar(
        x_data[1],
        plt_data_SGC[nx0 : nx0 + nx2],
        yerr=plt_err[nx0 + nx0 + nx2 : nx0 + nx2 + nx0 + nx2],
        marker="s",
        markerfacecolor=colours[1],
        markeredgecolor="k",
        color=colours[1],
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
        label=r"$\rm{SGC} \ P_{2}$"
    )    

    if pardict_NGC_SGC["do_hex"]:
        plt.errorbar(
            x_data[2],
            plt_data_NGC[nx0 + nx2 :],
            yerr=plt_err[nx0 + nx2 :],
            marker="o",
            markerfacecolor="g",
            markeredgecolor="k",
            color="g",
            linestyle="None",
            markeredgewidth=1.3,
            zorder=5,
        )

    plt.xlim(0.03, np.amax(pardict_NGC_SGC["xfit_max"]) * 1.05)
    plt.ylim(100, 2000)
    if pardict_NGC_SGC["do_corr"]:
        plt.xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$", fontsize=16)
        plt.ylabel(r"$s^{2}\xi(s)$", fontsize=16, labelpad=5)
    else:
        plt.xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$", fontsize=16)
        plt.ylabel(r"$kP(k)\,[h^{-2}\,\mathrm{Mpc}^{2}]$", fontsize=16, labelpad=5)
    plt.tick_params(width=1.3)
    plt.tick_params("both", length=10, which="major")
    plt.tick_params("both", length=5, which="minor")
    #plt.legend(prop={'size':12}, loc='best')
    plt.text(0.1395, 1850, r"$\rm{BOSS} \ \rm{DR12} \ z_{3}$", size=16)
    plt.text(0.14, 1720, r"$\rm{NGC} \ P_{0} \ \ \ \ \ \ \rm{SGC} \ P_{0}$", size=14)
    plt.scatter(0.1685, 1750, marker="o", color=colours[0])
    plt.scatter(0.2045, 1750, marker="o", color=colours[1]) 
    plt.text(0.14, 1590, r"$\rm{NGC} \ P_{2} \ \ \ \ \ \ \rm{SGC} \ P_{2}$", size=14)
    plt.scatter(0.1685, 1620, marker="s", color=colours[0])
    plt.scatter(0.2045, 1620, marker="s", color=colours[1]) 
    for axis in ["top", "left", "bottom", "right"]:
        plt.gca().spines[axis].set_linewidth(1.3)
    for tick in plt.gca().xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in plt.gca().yaxis.get_ticklabels():
        tick.set_fontsize(14)
    plt.tight_layout()
    plt.gca().set_autoscale_on(False)
    plt.ion()

    return plt

def create_plot(pardict, fittingdata):

    if pardict["do_hex"]:
        x_data = fittingdata.data["x_data"]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), len(x_data[2])
    else:
        x_data = fittingdata.data["x_data"][:2]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), 0
    fit_data = fittingdata.data["fit_data"]
    cov = fittingdata.data["cov"]

    plt_data = (
        np.concatenate(x_data) ** 2 * fit_data if pardict["do_corr"] else np.concatenate(x_data) ** 1.0 * fit_data
    )
    if pardict["do_corr"]:
        plt_err = np.concatenate(x_data) ** 2 * np.sqrt(cov[np.diag_indices(nx0 + nx2 + nx4)])
    else:
        plt_err = np.concatenate(x_data) ** 1.0 * np.sqrt(cov[np.diag_indices(nx0 + nx2 + nx0 + nx2)])

    plt.errorbar(
        x_data[0],
        plt_data[:nx0],
        yerr=plt_err[:nx0],
        marker="o",
        markerfacecolor="r",
        markeredgecolor="k",
        color="r",
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )
    plt.errorbar(
        x_data[1],
        plt_data[nx0 : nx0 + nx2],
        yerr=plt_err[nx0 : nx0 + nx2],
        marker="o",
        markerfacecolor="b",
        markeredgecolor="k",
        color="b",
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )
    if pardict["do_hex"]:
        plt.errorbar(
            x_data[2],
            plt_data[nx0 + nx2 :],
            yerr=plt_err[nx0 + nx2 :],
            marker="o",
            markerfacecolor="g",
            markeredgecolor="k",
            color="g",
            linestyle="None",
            markeredgewidth=1.3,
            zorder=5,
        )

    plt.xlim(0.03, np.amax(pardict["xfit_max"]) * 1.05)
    plt.ylim(100.0, 2000.0)
    if pardict["do_corr"]:
        plt.xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$", fontsize=16)
        plt.ylabel(r"$s^{2}\xi(s)$", fontsize=16, labelpad=5)
    else:
        plt.xlabel(r"$k\,(h\,\mathrm{Mpc}^{-1})$", fontsize=16)
        plt.ylabel(r"$kP(k)\,(h^{-2}\,\mathrm{Mpc}^{2})$", fontsize=16, labelpad=5)
    plt.tick_params(width=1.3)
    plt.tick_params("both", length=10, which="major")
    plt.tick_params("both", length=5, which="minor")
    for axis in ["top", "left", "bottom", "right"]:
        plt.gca().spines[axis].set_linewidth(1.3)
    for tick in plt.gca().xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in plt.gca().yaxis.get_ticklabels():
        tick.set_fontsize(14)
    plt.tight_layout()
    plt.gca().set_autoscale_on(False)
    plt.ion()

    return plt

def update_plot_combined(pardict, x_data, P_model_NGC, P_model_SGC, colours, fig_name, plt, keep=False): #runs NGC/SGC combined plots
    if pardict["do_hex"]:
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), len(x_data[2])
    else:
        x_data = x_data[:2]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), 0
    plt_data_NGC = np.concatenate(x_data) ** 2 * P_model if pardict["do_corr"] else np.concatenate(x_data) ** 1.0 * P_model_NGC
    plt_data_SGC = np.concatenate(x_data) ** 2 * P_model if pardict["do_corr"] else np.concatenate(x_data) ** 1.0 * P_model_SGC

    plt10 = plt.errorbar(
        x_data[0], plt_data_NGC[:nx0], marker="None", color="b", linestyle="-", markeredgewidth=1.3, zorder=0,
    )
    plt11 = plt.errorbar(
        x_data[1], plt_data_NGC[nx0 : nx0 + nx2], marker="None", color="b", linestyle="-", markeredgewidth=1.3, zorder=0,
    )
    plt10_SGC = plt.errorbar(
        x_data[0], plt_data_SGC[:nx0], marker="None", color="r", linestyle="-", markeredgewidth=1.3, zorder=0,
    )
    plt11_SGC = plt.errorbar(
        x_data[1], plt_data_SGC[nx0 : nx0 + nx2], marker="None", color="r", linestyle="-", markeredgewidth=1.3, zorder=0,
    )
    if pardict["do_hex"]:
        plt12 = plt.errorbar(
            x_data[2], plt_data[nx0 + nx2 :], marker="None", color="g", linestyle="-", markeredgewidth=1.3, zorder=0,
        )

    if keep:
        plt.ioff()
        plt.savefig(fig_name, dpi=300)
        plt.show()
    if not keep:
        plt.pause(0.005)
        if plt10 is not None:
            plt10.remove()
            plt10_SGC.remove()
        if plt11 is not None:
            plt11.remove()
            plt11_SGC.remove()
        if pardict["do_hex"]:
            if plt12 is not None:
                plt12.remove()

def update_plot_NGC_SGC_z1_z3(pardict_z1, pardict_z3, x_data_z1, x_data_z3, P_model_NGC_z1, P_model_SGC_z1, P_model_NGC_z3, P_model_SGC_z3, colours, fig_name, plt, keep=False): #code to run NGC/SGC z1+z3 combined plots
    if pardict_z1["do_hex"]:
        nx0_z1, nx2_z1, nx4_z1 = len(x_data_z1[0]), len(x_data_z1[1]), len(x_data_z1[2])
    else:
        x_data_z1 = x_data_z1[:2]
        nx0_z1, nx2_z1, nx4_z1 = len(x_data_z1[0]), len(x_data_z1[1]), 0

    if pardict_z3["do_hex"]:
        nx0_z3, nx2_z3, nx4_z3 = len(x_data_z3[0]), len(x_data_z3[1]), len(x_data_z3[2])
    else:
        x_data_z3 = x_data_z3[:2]
        nx0_z3, nx2_z3, nx4_z3 = len(x_data_z3[0]), len(x_data_z3[1]), 0

    plt_data_NGC_z1 = np.concatenate(x_data_z1) ** 2 * P_model if pardict_z1["do_corr"] else np.concatenate(x_data_z1) ** 1.0 * P_model_NGC_z1
    plt_data_SGC_z1 = np.concatenate(x_data_z1) ** 2 * P_model if pardict_z1["do_corr"] else np.concatenate(x_data_z1) ** 1.0 * P_model_SGC_z1

    plt_data_NGC_z3 = np.concatenate(x_data_z3) ** 2 * P_model if pardict_z3["do_corr"] else np.concatenate(x_data_z3) ** 1.0 * P_model_NGC_z3
    plt_data_SGC_z3 = np.concatenate(x_data_z3) ** 2 * P_model if pardict_z3["do_corr"] else np.concatenate(x_data_z3) ** 1.0 * P_model_SGC_z3

    plt_NGC_z1_mono = plt.errorbar(
        x_data_z1[0], plt_data_NGC_z1[:nx0_z1], marker="None", color=colours[0], linestyle="-", markeredgewidth=1.3, zorder=0,
    )
    plt_NGC_z1_quad = plt.errorbar(
        x_data_z1[1], plt_data_NGC_z1[nx0_z1 : nx0_z1 + nx2_z1], marker="None", color=colours[0], linestyle="-", markeredgewidth=1.3, zorder=0,
    )
    plt_SGC_z1_mono = plt.errorbar(
        x_data_z1[0], plt_data_SGC_z1[:nx0_z1], marker="None", color=colours[1], linestyle="-", markeredgewidth=1.3, zorder=0,
    )
    plt_SGC_z1_quad = plt.errorbar(
        x_data_z1[1], plt_data_SGC_z1[nx0_z1 : nx0_z1 + nx2_z1], marker="None", color=colours[1], linestyle="-", markeredgewidth=1.3, zorder=0,
    )

    plt_NGC_z3_mono = plt.errorbar(
        x_data_z3[0], plt_data_NGC_z3[:nx0_z3], marker="None", color=colours[2], linestyle="-", markeredgewidth=1.3, zorder=0,
    )
    plt_NGC_z3_quad = plt.errorbar(
        x_data_z3[1], plt_data_NGC_z3[nx0_z3 : nx0_z3 + nx2_z3], marker="None", color=colours[2], linestyle="-", markeredgewidth=1.3, zorder=0,
    )
    plt_SGC_z3_mono = plt.errorbar(
        x_data_z3[0], plt_data_SGC_z3[:nx0_z3], marker="None", color=colours[3], linestyle="-", markeredgewidth=1.3, zorder=0,
    )
    plt_SGC_z3_quad = plt.errorbar(
        x_data_z3[1], plt_data_SGC_z3[nx0_z3 : nx0_z3 + nx2_z3], marker="None", color=colours[3], linestyle="-", markeredgewidth=1.3, zorder=0,
    )

    if keep:
        plt.ioff()
        plt.show()
    if not keep:
        plt.pause(0.005)
        if plt_NGC_z1_mono is not None:
            plt_NGC_z1_mono.remove()
            plt_SGC_z1_mono.remove()
        if plt_NGC_z1_quad is not None:
            plt_NGC_z1_quad.remove()
            plt_SGC_z1_quad.remove()
        if plt_NGC_z3_mono is not None:
            plt_NGC_z3_mono.remove()
            plt_SGC_z3_mono.remove()
        if plt_NGC_z3_quad is not None:
            plt_NGC_z3_quad.remove()
            plt_SGC_z3_quad.remove()

def update_plot_individual(pardict, x_data, P_model, plt, keep=False):

    if pardict["do_hex"]:
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), len(x_data[2])
    else:
        x_data = x_data[:2]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), 0
    plt_data = np.concatenate(x_data) ** 2 * P_model if pardict["do_corr"] else np.concatenate(x_data) ** 1.0 * P_model

    plt10 = plt.errorbar(
        x_data[0], plt_data[:nx0], marker="None", color="r", linestyle="-", markeredgewidth=1.3, zorder=0,
    )
    plt11 = plt.errorbar(
        x_data[1], plt_data[nx0 : nx0 + nx2], marker="None", color="b", linestyle="-", markeredgewidth=1.3, zorder=0,
    )
    if pardict["do_hex"]:
        plt12 = plt.errorbar(
            x_data[2], plt_data[nx0 + nx2 :], marker="None", color="g", linestyle="-", markeredgewidth=1.3, zorder=0,
        )

    if keep:
        plt.ioff()
        plt.show()
    if not keep:
        plt.pause(0.005)
        if plt10 is not None:
            plt10.remove()
        if plt11 is not None:
            plt11.remove()
        if pardict["do_hex"]:
            if plt12 is not None:
                plt12.remove()


def update_plot_components(pardict, kin, P_components, plt, keep=False, comp_list=(True, True, True, True)):

    ls = [":", "-.", "--", "-"]
    labels = ["Linear", "Linear+Loop", "Linear+Loop+Counter", "Linear+Loop+Counter+Stoch"]
    kinfac = kin ** 2 if pardict["do_corr"] else kin ** 1.0

    part_comp = [np.zeros(len(kin)), np.zeros(len(kin)), np.zeros(len(kin))]
    for (line, comp, add, label) in zip(ls, P_components, comp_list, labels):
        for i, c in enumerate(comp):
            part_comp[i] += c
        if add:
            plt10 = plt.errorbar(
                kin,
                kinfac * part_comp[0],
                marker="None",
                color="r",
                linestyle=line,
                markeredgewidth=1.3,
                zorder=0,
                label=label,
            )
            plt11 = plt.errorbar(
                kin, kinfac * part_comp[1], marker="None", color="b", linestyle=line, markeredgewidth=1.3, zorder=0,
            )
            if pardict["do_hex"]:
                plt12 = plt.errorbar(
                    kin, kinfac * part_comp[2], marker="None", color="g", linestyle=line, markeredgewidth=1.3, zorder=0,
                )
    plt.legend()

    if keep:
        plt.ioff()
        plt.show()
    if not keep:
        plt.pause(0.005)
        if plt10 is not None:
            plt10.remove()
        if plt11 is not None:
            plt11.remove()
        if pardict["do_hex"]:
            if plt12 is not None:
                plt12.remove()


def format_pardict(pardict):

    pardict["do_corr"] = int(pardict["do_corr"])
    pardict["do_marg"] = int(pardict["do_marg"])
    pardict["do_hex"] = int(pardict["do_hex"])
    pardict["taylor_order"] = int(pardict["taylor_order"])
    pardict["xfit_min"] = np.array(pardict["xfit_min"]).astype(float)
    pardict["xfit_max"] = np.array(pardict["xfit_max"]).astype(float)
    pardict["order"] = int(pardict["order"])
    pardict["template_order"] = int(pardict["template_order"])

    return pardict


def do_optimization(func, start, birdmodel, fittingdata, plt):

    from scipy.optimize import basinhopping

    result = basinhopping(
        func,
        start,
        niter_success=10,
        niter=100,
        stepsize=0.01,
        minimizer_kwargs={
            "args": (birdmodel, fittingdata, plt),
            "method": "Nelder-Mead",
            "tol": 1.0e-4,
            "options": {"maxiter": 40000, "xatol": 1.0e-4, "fatol": 1.0e-4},
        },
    )
    print("#-------------- Best-fit----------------")
    print(result)

    return result


def read_chain(chainfile, burnlimitlow=5000, burnlimitup=None):

    # Read in the samples
    walkers = []
    samples = []
    like = []
    infile = open(chainfile, "r")
    for line in infile:
        ln = line.split()
        samples.append(list(map(float, ln[1:-1])))
        walkers.append(int(ln[0]))
        like.append(float(ln[-1]))
    infile.close()

    like = np.array(like)
    walkers = np.array(walkers)
    samples = np.array(samples)
    nwalkers = max(walkers)

    if burnlimitup is None:
        bestid = np.argmax(like)
    else:
        bestid = np.argmax(like[: np.amax(walkers) * burnlimitup])

    burntin = []
    burntlike = []
    nburntin = 0

    for i in range(nwalkers + 1):
        ind = np.where(walkers == i)[0]
        if len(ind) == 0:
            continue
        x = [j for j in range(len(ind))]
        if burnlimitup is None:
            ind2 = np.where(np.asarray(x) >= burnlimitlow)[0]
        else:
            ind2 = np.where(np.logical_and(np.asarray(x) >= burnlimitlow, np.asarray(x) <= burnlimitup))[0]
        for k in range(len(ind2 + 1)):
            burntin.append(samples[ind[ind2[k]]])
            burntlike.append(like[ind[ind2[k]]])
        nburntin += len(ind2)
    burntin = np.array(burntin)
    burntlike = np.array(burntlike)

    return burntin, samples[bestid], burntlike


def read_chain_backend(chainfile):

    import copy
    import emcee

    reader = emcee.backends.HDFBackend(chainfile)

    tau = reader.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    samples = reader.get_chain(discard=burnin, flat=True)
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True)
    bestid = np.argmax(log_prob_samples)

    return samples, copy.copy(samples[bestid]), log_prob_samples
