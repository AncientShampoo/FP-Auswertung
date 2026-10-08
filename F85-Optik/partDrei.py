import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import maybeNeeded
from scipy.optimize import curve_fit

# PATH_TO_FILE = pathlib.Path(os.path.realpath(__file__))
BASE_DIR = pathlib.Path(__file__).parent
IMAGES = BASE_DIR / "Images" / "partZwei"
print(IMAGES)

##################################################################
# # --------------------- Part 3.2.2  ------------------------ # #
# Frequenzabhängigkeit Abstand Minima erster und zweiter Ordnung #
##################################################################


### ------- Diffraction order distances for different frequencies ------ ###

dist_second_ord_2_2 = np.array([8.7, 9.4, 10.1, 10.7, 11.3])
dist_second_ord_0_2 = np.array([6.0, np.nan])
data_diffraction_distances = {
    "Frequency": np.array([92.4, 99.8, 106.5, 112.5, 118.8, 124.8, 131.0]),
    "Frequ_err": np.array([2.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.0]),
    "dist_first_ord": np.array([4.4, 4.7, 5.0, 5.3, 5.7, 5.9, 6.2]),
    "dist_err_first": np.ones(len(dist_second_ord_2_2) + len(dist_second_ord_0_2)),
    "dist_second_ord": np.concatenate((dist_second_ord_2_2, dist_second_ord_0_2 * 2)),
    "dist_err_second": np.ones(len(dist_second_ord_2_2) + len(dist_second_ord_0_2)),
}
data_diffraction_distances["dist_err_second"][-1] = (
    4  # doubling error since only half way measurement
)
pd_diffract_dist = pd.DataFrame(data_diffraction_distances)
print(pd_diffract_dist)

plt.errorbar(
    x=pd_diffract_dist["Frequency"],
    y=pd_diffract_dist["dist_first_ord"],
    xerr=pd_diffract_dist["Frequ_err"],
    yerr=pd_diffract_dist["dist_err_first"],
    fmt="o",
)
plt.errorbar(
    x=pd_diffract_dist["Frequency"],
    y=pd_diffract_dist["dist_second_ord"],
    xerr=pd_diffract_dist["Frequ_err"],
    yerr=pd_diffract_dist["dist_err_second"],
    fmt="o",
)
# plt.show()
plt.savefig(IMAGES / "part_3.1.1_fitt_frequecyVsPowerBOth.jpg", dpi=900)

plt.close()

### ------ Sound velocity ------ ###


phasgesch, phas_err = maybeNeeded.phasenGeschwMitErr(
        f_Ultrasch=pd_diffract_dist["Frequency"][:-1] * 10**6,
        f_err=pd_diffract_dist["Frequ_err"][:-1] * 10**6,
        d_Abst=pd_diffract_dist["dist_first_ord"][:-1] * 10**-2 / 2,
        d_err=pd_diffract_dist["dist_err_first"][:-1] * 10**-3 / 2,
        N_beugOrd=1
    )
print(f"Schallgeschwindigkeit mit fehler über 1.Ordnung:")
for i in range(len(phasgesch)):
    print(phasgesch[i], "+/-", phas_err[i])
print()


phasgesch, phas_err = maybeNeeded.phasenGeschwMitErr(
    f_Ultrasch=pd_diffract_dist["Frequency"][:-1] * 10**6,
    f_err=pd_diffract_dist["Frequ_err"][:-1] * 10**6,
    d_Abst=pd_diffract_dist["dist_second_ord"][:-1] * 10**-2 / 2,
    d_err=pd_diffract_dist["dist_err_second"][:-1] * 10**-3 / 2,
    N_beugOrd=2
)
print(f"Schallgeschwindigkeit mit fehler über 2.Ordnung:")
for i in range(len(phasgesch)):
    print(phasgesch[i], "+/-", phas_err[i])
print()

print(f"Mittelwert Schall geschw über 1. Ord:")
print(f"")

### ------ Frequency depandency of transmitted power ------ ###

data_frequency_power = {
    "Frequency": np.array([92.4, 99.8, 106.5, 112.5, 118.8, 124.8, 131.0]),
    "Frequ_err": np.array([2, 1, 1.5, 1.5, 1.5, 2, 2]),
    "Power_zeroth_ord": np.array([538, 667, 702, 680, 670, 635, 625]),
    "Power_zero_err": np.array([1.9, 6.0, 2.5, 5, 10, 10, 10]),
    "Power_first_ord": np.array([3.5, 4, 3.5, 3.5, 6.0, 17.0, 12.3]),
    "Power_first_err": np.array([1, 1, 1, 1, 1.5, 1.5, 1.5]),
}

pd_frequency_power = pd.DataFrame(data_frequency_power)
print(pd_frequency_power)

plt.errorbar(
    x=pd_frequency_power["Frequency"],
    y=pd_frequency_power["Power_zeroth_ord"],
    xerr=pd_frequency_power["Frequ_err"],
    yerr=pd_frequency_power["Power_zero_err"],
    fmt="o",
)

plt.errorbar(
    x=pd_frequency_power["Frequency"],
    y=pd_frequency_power["Power_first_ord"],
    xerr=pd_frequency_power["Frequ_err"],
    yerr=pd_frequency_power["Power_first_err"],
    fmt="o",
)
plt.savefig(IMAGES / "part_3.1.1_fitt_frequecyVsPower.jpg", dpi=900)

# plt.show()
plt.close()
