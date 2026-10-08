import numpy as np

lambd_optWavelength_nm = 632.8              # nm
lambd_optWavelength_m = 632.8 * 10**(-9)    # m
L_crystalLength_mm = 20.0                   # mm
L_crystalLength_m = 20.0 * 10**(-3)         # m
n0_indicesRefraction = 2.261
ne_indicesRefraction = 2.142
M2_figOfMerit = 34.5 * 10**(-15)            # s^3 kg^-1

pockels_coeffs = np.array(
    [   [0,     -3.4,   8.6 ],
        [0,     3.4,    8.6 ],
        [0,     0,      30.8],
        [0,     28,     0   ],
        [28,    0,      0   ],
        [-3.4,  0,      0   ]]
) # Not reliable, most defenitely not to use


#### Functions ####
def R_powerTransmittedToFirstOrder(
        theta:float,
        I_s: float,
        L: float = L_crystalLength_m,
        lambd:float = lambd_optWavelength_m,
        M2: float = M2_figOfMerit,
    ):
    """something"""
    R = np.sin(((np.pi*L)*np.sqrt((M2*I_s)/2)) / (lambd*np.cos(theta)))**2
    return R

def phasengeschwKristall(
        f_Ultrasch,
        d_Abst,
        N_beugOrd,
        lam_Licht = 632.8 * 10**-9,
        x_Schirm = 1550 * 10**-3,
):
    """
    f_Ultrasch: in Hz
    d_Abst: Abstand der Beugungsmaxima zur Nullten Ordnung
    x_Schirm: Abstand Schirm in m
    """
    c_s = ((x_Schirm * N_beugOrd * lam_Licht) / (d_Abst) ) * f_Ultrasch
    return c_s

def phasenGeschwMitErr(
    f_Ultrasch,
    f_err,
    d_Abst,
    d_err,
    N_beugOrd,
    x_Schirm = 1550 * 10**-3,
    x_Schirm_err = 4 * 10**-3,
    lam_Licht = 632.8 * 10**-9,
):
    c_s = ((x_Schirm * N_beugOrd * lam_Licht) / (d_Abst) ) * f_Ultrasch
    dc_dx = np.abs(N_beugOrd * lam_Licht * f_Ultrasch / d_Abst)
    dc_df = np.abs(N_beugOrd * lam_Licht * x_Schirm / d_Abst)
    dc_dd = np.abs(-N_beugOrd * lam_Licht * f_Ultrasch * x_Schirm / d_Abst**2)
    c_s_err = np.sqrt((dc_df * f_err)**2 + (dc_dd * d_err)**2 + (dc_dx * x_Schirm_err)**2)
    return (np.round(c_s, 3), np.round(c_s_err, 3))

def phasenGeschwMitErrMittelwerte(
    f_Ultrasch,
    f_err,
    d_Abst,
    d_err,
    N_beugOrd,
    x_Schirm = 1550 * 10**-3,
    x_Schirm_err = 4 * 10**-3,
    lam_Licht = 632.8 * 10**-9,
):
    c_s = ((x_Schirm * N_beugOrd * lam_Licht) / (d_Abst) ) * f_Ultrasch
    c_s_mittelwert = np.mean(c_s)
    dc_dx = np.abs(N_beugOrd * lam_Licht * f_Ultrasch / d_Abst)
    dc_df = np.abs(N_beugOrd * lam_Licht * x_Schirm / d_Abst)
    dc_dd = np.abs(-N_beugOrd * lam_Licht * f_Ultrasch * x_Schirm / d_Abst**2)
    c_s_err = dc_df * f_err + dc_dd * d_err + dc_dx * x_Schirm_err
    return (np.round(c_s, 3), np.round(c_s_err, 3))
