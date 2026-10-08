import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path
# from chshUngl import data_alice

BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "FP66-quant" / "2026-07-20"
DATA_KORR_PATH = DATA_PATH / "KorrelationsMessungen"
DATA_RATEN_PATH = DATA_PATH / "RatenMessungen"
DATA_BIT_PATH = DATA_PATH / "BitMessungen"

IMAGE_PATH = BASE_DIR / "Images"
EINZELZ_PATH = IMAGE_PATH / 'Einzelzählraten'
KOINZIDENZ_PATH = IMAGE_PATH / 'Koinzidenz'


### ---- Matplotlib settings ---- ###
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['text.usetex'] = True

FONTSIZE = 14

### ---- Utility Functions ---- ###
ROUND_TO = 4
def get_unsicherheit(df, key):
    N_max = np.max(df[key])
    N_min = np.min(df[key])
    return (np.round(np.sqrt(N_min), ROUND_TO), np.round(np.sqrt(N_max), ROUND_TO))

def get_polaristaoinsgrad(df, key, range_start=0, range_stop=-1):
    N_max = np.max(df[key][range_start:range_stop])
    N_min = np.min(df[key][range_start:range_stop])
    return np.round((N_max - N_min) / (N_max + N_min), ROUND_TO)

def get_unsicherheit_polarisationsgrad(df, key):
    N_max = np.max(df[key])
    N_min = np.min(df[key])
    return np.round((2 * np.sqrt(N_max * N_min / (N_max + N_min)**3)), ROUND_TO)


### ---- Curve fitting Functions ---- ###
def func_sin_fitt(x, a, b, c, d):
    return a * np.sin(b * (x + c)) + d

def estimate_sin_p0(x, y, n_points=None):
    """Estimate [a, b, c, d] starting values for a*sin(b*(x+c))+d
    from the max/min within the first n_points of (x, y)."""
    x = np.asarray(x)
    y = np.asarray(y)
    if n_points is not None:
        x, y = x[:n_points], y[:n_points]

    idx_max = np.argmax(y)
    idx_min = np.argmin(y)
    x_max, x_min = x[idx_max], x[idx_min]
    y_max, y_min = y[idx_max], y[idx_min]

    print(x_max, x_min, y_max, y_min)

    a0 = (y_max - y_min) / 2
    b0 = np.pi / np.abs((x_max - x_min))
    c0 = -(x_max + x_min) / 2   # note the sign fix
    d0 = (y_max + y_min) / 2

    return [a0, b0, c0, d0]


err_DEGREE = 0.5

columns = [
    'AngleAlice(deg)',
    'AngleBob(deg)',
    'A0',
    'A1',
    'B0',
    'B1',
    'A0B0',
    'A1B0',
    'A0B1',
    'A1B1',
]

data_bob = pd.read_csv(
    filepath_or_buffer = DATA_PATH / "RatenMessungen/2026-07-20_13-56-28_Zufall_Bob.txt",
    sep     = ';',
    comment = '#',
    header  = None,
    names   = columns
)
data_bob.sort_values('AngleBob(deg)', inplace=True, ignore_index=True)
data_alice = pd.read_csv(
    filepath_or_buffer = DATA_PATH / "RatenMessungen/2026-07-20_14-06-37_Zufall_Alice.txt",
    sep     = ';',
    comment = '#',
    header  = None,
    names   = columns
)
data_alice.sort_values('AngleAlice(deg)', inplace=True, ignore_index=True)
# print(data)

print(
    f"Polaristaionsgrad Bob 0:  {get_polaristaoinsgrad(data_bob, 'B0')} +/- {get_unsicherheit_polarisationsgrad(data_bob, 'B0')}",
    f"nmin: {np.min(data_bob['B0'])} pm {get_unsicherheit(data_bob, 'B0')[0]}",
    f"nmax: {np.max(data_bob['B0'])} pm {get_unsicherheit(data_bob, 'B0')[1]}"
)
print(
    f"Polaristaionsgrad Bob 0 nur stabiler bereich:  {get_polaristaoinsgrad(data_bob, 'B0', 18)} +/- {get_unsicherheit_polarisationsgrad(data_bob, 'B0')}",
    f"nmin: {np.min(data_bob['B0'][:    18])} pm {get_unsicherheit(data_bob, 'B0')[0]}",
    f"nmax: {np.max(data_bob['B0'][:18])} pm {get_unsicherheit(data_bob, 'B0')[1]}"
)
print(
    f"Polaristaionsgrad Bob 1:  {get_polaristaoinsgrad(data_bob, 'B1')} +/- {get_unsicherheit_polarisationsgrad(data_bob, 'B1')}",
    f"nmin: {np.min(data_bob['B1'])} pm {get_unsicherheit(data_bob, 'B1')[0]}",
    f"nmax: {np.max(data_bob['B1'])} pm {get_unsicherheit(data_bob, 'B1')[1]}"
)

print(
    f"Polaristaionsgrad Alice 0:  {get_polaristaoinsgrad(data_alice, 'A0')} +/- {get_unsicherheit_polarisationsgrad(data_alice, 'A0')}",
    f"nmin: {np.min(data_alice['A0'])} pm {get_unsicherheit(data_alice, 'A0')[0]}",
    f"nmax: {np.max(data_alice['A0'])} pm {get_unsicherheit(data_alice, 'A0')[1]}"
)
print(
    f"Polaristaionsgrad Alice 1:  {get_polaristaoinsgrad(data_alice, 'A1')} +/- {get_unsicherheit_polarisationsgrad(data_alice, 'A1')}",
    f"nmin: {np.min(data_alice['A1'])} pm {get_unsicherheit(data_alice, 'A1')[0]}",
    f"nmax: {np.max(data_alice['A1'])} pm {get_unsicherheit(data_alice, 'A1')[1]}"
)

x_values_fit = np.linspace(0, 180, 10000)

n_points=18
popt, pcov = curve_fit(func_sin_fitt, data_bob['AngleBob(deg)'][:n_points], data_bob['B1'][:n_points])

plt.scatter(data_bob['AngleBob(deg)'], data_bob['B1'])
plt.title(r'\textbf{Count rates without filtering for coincidences}')
plt.xlabel(r"Angle HWP $[^\circ]$", fontsize=FONTSIZE)
plt.ylabel(r"Counts $N$", fontsize=FONTSIZE)
plt.savefig(
    EINZELZ_PATH / 'EinzelzählratenEinzeln_Bob_1.jpg',
    dpi = 900,
    bbox_inches="tight"
)
plt.close()


n_points = 18
popt, pcov = curve_fit(
    func_sin_fitt,
    data_bob['AngleBob(deg)'][0:n_points],
    data_bob['B0'][0:n_points],
    p0=estimate_sin_p0(
            data_bob['AngleBob(deg)'],
            data_bob['B0'],
            n_points=n_points
        ),
    bounds=(
            [-np.inf,      2*np.pi/150, -np.inf, -np.inf],   # lower: a>0, period<150°
            [np.inf, 2*np.pi/60,   np.inf,  np.inf]    # upper: period>60°
        ),
    method='trf',
    max_nfev=6000
    )

plt.scatter(data_bob['AngleBob(deg)'], data_bob['B0'])
plt.plot(
    x_values_fit,
    func_sin_fitt(x_values_fit, *popt)
)
plt.title(r'\textbf{Count rates without filtering for coincidences}')
plt.xlabel(r"Angle HWP $[^\circ]$", fontsize=FONTSIZE)
plt.ylabel(r"Counts $N$", fontsize=FONTSIZE)
plt.savefig(
    EINZELZ_PATH / 'EinzelzählratenEinzeln_Bob_0.jpg',
    dpi = 900,
    bbox_inches="tight"
)
plt.close()

plt.scatter(data_bob['AngleBob(deg)'], data_bob['B0'])
plt.scatter(data_bob['AngleBob(deg)'], data_bob['B1'])
# plt.show()
plt.savefig(
    EINZELZ_PATH / 'EinzelzählratenBeideDetektoren_Bob.jpg',
    dpi = 900,
    bbox_inches="tight"
)
plt.close()

###########################################
## --------- Polar Coordinates --------- ##
plt.polar(data_bob['AngleBob(deg)'], data_bob['B1'], marker='*', linestyle='')
# plt.show()
plt.savefig(
    EINZELZ_PATH / 'EinzelzählratenEinzeln_Polar_Bob_1.jpg',
    dpi = 900,
    bbox_inches="tight"
)
plt.close()
plt.polar(data_bob['AngleBob(deg)'], data_bob['B0'], marker='*', linestyle='')
# plt.show()
plt.savefig(
    EINZELZ_PATH / 'EinzelzählratenEinzeln_Polar_Bob_0.jpg',
    dpi = 900,
    bbox_inches="tight"
)
plt.close()

plt.polar(data_bob['AngleBob(deg)'], data_bob['B0'], marker='*', linestyle='')
plt.polar(data_bob['AngleBob(deg)'], data_bob['B1'], marker='.', linestyle='')
# plt.show()
plt.savefig(
    EINZELZ_PATH / 'EinzelzählratenBeideDetektoren_Polar_Bob.jpg',
    dpi = 900,
    bbox_inches="tight"
)
plt.close()




plt.scatter(data_alice['AngleAlice(deg)'], data_alice['A0'])
plt.scatter(data_alice['AngleAlice(deg)'], data_alice['A1'])
# plt.show()
plt.savefig(
    EINZELZ_PATH / 'EinzelzählratenBeideDetektoren_Alice.jpg',
    dpi = 900,
    bbox_inches="tight"
)
plt.close()

plt.scatter(data_alice['AngleAlice(deg)'], data_alice['A0'])
plt.title(r'\textbf{Count rates without filtering for coincidences}')
plt.xlabel(r"Angle HWP $[^\circ]$", fontsize=FONTSIZE)
plt.ylabel(r"Counts $N$", fontsize=FONTSIZE)
plt.savefig(
    EINZELZ_PATH / 'EinzelzählratenEinzeln_Alice_0.jpg',
    dpi = 900,
    bbox_inches="tight"
)
plt.close()

plt.scatter(data_alice['AngleAlice(deg)'], data_alice['A1'])
plt.title(r'\textbf{Count rates without filtering for coincidences}')
plt.xlabel(r"Angle HWP $[^\circ]$", fontsize=FONTSIZE)
plt.ylabel(r"Counts $N$", fontsize=FONTSIZE)
plt.savefig(
    EINZELZ_PATH / 'EinzelzählratenEinzeln_Alice_1.jpg',
    dpi = 900,
    bbox_inches="tight"
)
plt.close()



###########################################
## --------------- Alice --------------- ##
## --------- Polar Coordinates --------- ##
# # # # # # # # # # # # # # # # # # # # # #

plt.polar(data_alice['AngleAlice(deg)'], data_alice['A1'], marker='*', linestyle='')
# plt.show()
plt.savefig(
    EINZELZ_PATH / 'EinzelzählratenEinzeln_Polar_Alice_1.jpg',
    dpi = 900,
    bbox_inches="tight"
)
plt.close()

plt.polar(data_alice['AngleAlice(deg)'], data_alice['A0'], marker='*', linestyle='')
# plt.show()
plt.savefig(
    EINZELZ_PATH / 'EinzelzählratenEinzeln_Polar_Alice_0.jpg',
    dpi = 900,
    bbox_inches="tight"
)
plt.close()

#################
# --- Beide --- #
#################
plt.polar(data_alice['AngleAlice(deg)'], data_alice['A0'], marker='*', linestyle='')
plt.polar(data_alice['AngleAlice(deg)'], data_alice['A1'], marker='.', linestyle='')
# plt.show()
plt.savefig(
    EINZELZ_PATH / 'EinzelzählratenBeideDetektoren_Polar_Alice.jpg',
    dpi = 900,
    bbox_inches="tight"
)
plt.close()




#########################################
# --------Korrelations messung--------- #
#########################################
# +45
DATA_NAME_ALICE = '2026-07-20_15-20-46_Korrelation_Alice.txt'
DATA_NAME_BOB = '2026-07-20_15-11-59_Korrelation_Bob.txt'
data_bob_corr = pd.read_csv(
    filepath_or_buffer = DATA_RATEN_PATH / DATA_NAME_BOB,
    sep     = ';',
    comment = '#',
    header  = None,
    names   = columns
)
data_alice_corr = pd.read_csv(
    filepath_or_buffer = DATA_RATEN_PATH / DATA_NAME_ALICE,
    sep     = ';',
    comment = '#',
    header  = None,
    names   = columns
)

print('Fixiert Alice +45, Bobs detektoren:')
print(
    f"Polaristaionsgrad Koinzidenz A0B0:  {get_polaristaoinsgrad(data_bob_corr, 'A0B0')} +/- {get_unsicherheit_polarisationsgrad(data_bob_corr, 'A0B0')}",
    f"nmin: {np.min(data_bob_corr['A0B0'])} pm {get_unsicherheit(data_bob_corr, 'A0B0')[0]}",
    f"nmax: {np.max(data_bob_corr['A0B0'])} pm {get_unsicherheit(data_bob_corr, 'A0B0')[1]}"
)
print(
    f"Polaristaionsgrad Koinzidenz A0B1:  {get_polaristaoinsgrad(data_bob_corr, 'A0B1')} +/- {get_unsicherheit_polarisationsgrad(data_bob_corr, 'A0B1')}",
    f"nmin: {np.min(data_bob_corr['A0B1'])} pm {get_unsicherheit(data_bob_corr, 'A0B1')[0]}",
    f"nmax: {np.max(data_bob_corr['A0B1'])} pm {get_unsicherheit(data_bob_corr, 'A0B1')[1]}"
)
print(
    f"Polaristaionsgrad Koinzidenz A1B0:  {get_polaristaoinsgrad(data_bob_corr, 'A1B0')} +/- {get_unsicherheit_polarisationsgrad(data_bob_corr, 'A1B0')}",
    f"nmin: {np.min(data_bob_corr['A1B0'])} pm {get_unsicherheit(data_bob_corr, 'A1B0')[0]}",
    f"nmax: {np.max(data_bob_corr['A1B0'])} pm {get_unsicherheit(data_bob_corr, 'A1B0')[1]}"
)
print(
    f"Polaristaionsgrad Koinzidenz A1B1:  {get_polaristaoinsgrad(data_bob_corr, 'A1B1')} +/- {get_unsicherheit_polarisationsgrad(data_bob_corr, 'A1B1')}",
    f"nmin: {np.min(data_bob_corr['A1B1'])} pm {get_unsicherheit(data_bob_corr, 'A1B1')[0]}",
    f"nmax: {np.max(data_bob_corr['A1B1'])} pm {get_unsicherheit(data_bob_corr, 'A1B1')[1]}"
)
print()


print('Fixiert Bob +45, Alices Detektoren:')
print(
    f"Polaristaionsgrad Koinzidenz A0B0:  {get_polaristaoinsgrad(data_alice_corr, 'A0B0')} +/- {get_unsicherheit_polarisationsgrad(data_alice_corr, 'A0B0')}",
    f"nmin: {np.min(data_alice_corr['A0B0'])} pm {get_unsicherheit(data_alice_corr, 'A0B0')[0]}",
    f"nmax: {np.max(data_alice_corr['A0B0'])} pm {get_unsicherheit(data_alice_corr, 'A0B0')[1]}"
)
print(
    f"Polaristaionsgrad Koinzidenz A0B1:  {get_polaristaoinsgrad(data_alice_corr, 'A0B1')} +/- {get_unsicherheit_polarisationsgrad(data_alice_corr, 'A0B1')}",
    f"nmin: {np.min(data_alice_corr['A0B1'])} pm {get_unsicherheit(data_alice_corr, 'A0B1')[0]}",
    f"nmax: {np.max(data_alice_corr['A0B1'])} pm {get_unsicherheit(data_alice_corr, 'A0B1')[1]}"
)
print(
    f"Polaristaionsgrad Koinzidenz A1B0:  {get_polaristaoinsgrad(data_alice_corr, 'A1B0')} +/- {get_unsicherheit_polarisationsgrad(data_alice_corr, 'A1B0')}",
    f"nmin: {np.min(data_alice_corr['A1B0'])} pm {get_unsicherheit(data_alice_corr, 'A1B0')[0]}",
    f"nmax: {np.max(data_alice_corr['A1B0'])} pm {get_unsicherheit(data_alice_corr, 'A1B0')[1]}"
)
print(
    f"Polaristaionsgrad Koinzidenz A1B1:  {get_polaristaoinsgrad(data_alice_corr, 'A1B1')} +/- {get_unsicherheit_polarisationsgrad(data_alice_corr, 'A1B1')}",
    f"nmin: {np.min(data_alice_corr['A1B1'])} pm {get_unsicherheit(data_alice_corr, 'A1B1')[0]}",
    f"nmax: {np.max(data_alice_corr['A1B1'])} pm {get_unsicherheit(data_alice_corr, 'A1B1')[1]}"
)
print()
plt.scatter(data_bob_corr['AngleBob(deg)'], data_bob_corr['A0B0'])
plt.scatter(data_bob_corr['AngleBob(deg)'], data_bob_corr['A0B1'])
plt.title(r'\textbf{Coincidence count rate}' + '\n' + r' Alice HWP fixated at $+45^\circ$')
# plt.show()
plt.xlabel(r"Angle HWP $[^\circ]$", fontsize=FONTSIZE)
plt.ylabel(r"Counts $N$", fontsize=FONTSIZE)
plt.savefig(
    KOINZIDENZ_PATH / 'KoinzidenztählratenAliceFixiert45.jpg',
    dpi=900,
    bbox_inches="tight"
)
plt.close()

plt.scatter(data_alice_corr['AngleAlice(deg)'], data_alice_corr['A0B0'])
plt.scatter(data_alice_corr['AngleAlice(deg)'], data_alice_corr['A1B0'])
plt.title(r'\textbf{Coincidence count rate}' + '\n' + r' Bob HWP fixated at $+45^\circ$')
plt.xlabel(r"Angle HWP $[^\circ]$", fontsize=FONTSIZE)
plt.ylabel(r"Counts $N$", fontsize=FONTSIZE)

# plt.show()
plt.savefig(
    KOINZIDENZ_PATH / 'KoinzidenztählratenBobFixiert45.jpg',
    dpi=900,
    bbox_inches="tight"
)
plt.close()




# ---------------0/90--------------- #
DATA_NAME_ALICE = '2026-07-20_14-58-07_Korrelation_Alice.txt'
DATA_NAME_BOB = '2026-07-20_14-42-25_Korrelation_Bob.txt'
data_bob_corr = pd.read_csv(
    filepath_or_buffer = DATA_RATEN_PATH / DATA_NAME_BOB,
    sep     = ';',
    comment = '#',
    header  = None,
    names   = columns
)
data_alice_corr = pd.read_csv(
    filepath_or_buffer = DATA_RATEN_PATH / DATA_NAME_ALICE,
    sep     = ';',
    comment = '#',
    header  = None,
    names   = columns
)
print("Fixiert Alice 0/90 grad, Bobs Detektor")
# print('Polaristaionsgrad Koinzidenz A0B0: ', get_polaristaoinsgrad(data_bob_corr, 'A0B0'))
# print('Polaristaionsgrad Koinzidenz A0B1: ', get_polaristaoinsgrad(data_bob_corr, 'A0B1'))
# print('Polaristaionsgrad Koinzidenz A1B0: ', get_polaristaoinsgrad(data_bob_corr, 'A1B0'))
# print('Polaristaionsgrad Koinzidenz A1B1: ', get_polaristaoinsgrad(data_bob_corr, 'A1B1'))
print(
    f"Polaristaionsgrad Koinzidenz A0B0:  {get_polaristaoinsgrad(data_bob_corr, 'A0B0')} +/- {get_unsicherheit_polarisationsgrad(data_alice_corr, 'A0B0')}",
    f"nmin: {np.min(data_bob_corr['A0B0'])} pm {get_unsicherheit(data_bob_corr, 'A0B0')[0]}",
    f"nmax: {np.max(data_bob_corr['A0B0'])} pm {get_unsicherheit(data_bob_corr, 'A0B0')[1]}"
)
print(
    f"Polaristaionsgrad Koinzidenz A0B1:  {get_polaristaoinsgrad(data_bob_corr, 'A0B1')} +/- {get_unsicherheit_polarisationsgrad(data_alice_corr, 'A0B1')}",
    f"nmin: {np.min(data_bob_corr['A0B1'])} pm {get_unsicherheit(data_bob_corr, 'A0B1')[0]}",
    f"nmax: {np.max(data_bob_corr['A0B1'])} pm {get_unsicherheit(data_bob_corr, 'A0B1')[1]}"
)
print(
    f"Polaristaionsgrad Koinzidenz A1B0:  {get_polaristaoinsgrad(data_bob_corr, 'A1B0')} +/- {get_unsicherheit_polarisationsgrad(data_alice_corr, 'A1B0')}",
    f"nmin: {np.min(data_bob_corr['A1B0'])} pm {get_unsicherheit(data_bob_corr, 'A1B0')[0]}",
    f"nmax: {np.max(data_bob_corr['A1B0'])} pm {get_unsicherheit(data_bob_corr, 'A1B0')[1]}"
)
print(
    f"Polaristaionsgrad Koinzidenz A1B1:  {get_polaristaoinsgrad(data_bob_corr, 'A1B1')} +/- {get_unsicherheit_polarisationsgrad(data_alice_corr, 'A1B1')}",
    f"nmin: {np.min(data_bob_corr['A1B1'])} pm {get_unsicherheit(data_bob_corr, 'A1B1')[0]}",
    f"nmax: {np.max(data_bob_corr['A1B1'])} pm {get_unsicherheit(data_bob_corr, 'A1B1')[1]}"
)
print()
print()

print("Fixiert Bob 0/90 grad, Alices Detektor")
# print('Polaristaionsgrad Koinzidenz A0B0: ', get_polaristaoinsgrad(data_alice_corr, 'A0B0'))
# print('Polaristaionsgrad Koinzidenz A0B1: ', get_polaristaoinsgrad(data_alice_corr, 'A0B1'))
# print('Polaristaionsgrad Koinzidenz A1B0: ', get_polaristaoinsgrad(data_alice_corr, 'A1B0'))
# print('Polaristaionsgrad Koinzidenz A1B1: ', get_polaristaoinsgrad(data_alice_corr, 'A1B1'))
print(
    f"Polaristaionsgrad Koinzidenz A0B0:  {get_polaristaoinsgrad(data_alice_corr, 'A0B0')} +/- {get_unsicherheit_polarisationsgrad(data_alice_corr, 'A0B0')}",
    f"nmin: {np.min(data_alice_corr['A0B0'])} pm {get_unsicherheit(data_alice_corr, 'A0B0')[0]}",
    f"nmax: {np.max(data_alice_corr['A0B0'])} pm {get_unsicherheit(data_alice_corr, 'A0B0')[1]}"
)
print(
    f"Polaristaionsgrad Koinzidenz A0B1:  {get_polaristaoinsgrad(data_alice_corr, 'A0B1')} +/- {get_unsicherheit_polarisationsgrad(data_alice_corr, 'A0B1')}",
    f"nmin: {np.min(data_alice_corr['A0B1'])} pm {get_unsicherheit(data_alice_corr, 'A0B1')[0]}",
    f"nmax: {np.max(data_alice_corr['A0B1'])} pm {get_unsicherheit(data_alice_corr, 'A0B1')[1]}"
)
print(
    f"Polaristaionsgrad Koinzidenz A1B0:  {get_polaristaoinsgrad(data_alice_corr, 'A1B0')} +/- {get_unsicherheit_polarisationsgrad(data_alice_corr, 'A1B0')}",
    f"nmin: {np.min(data_alice_corr['A1B0'])} pm {get_unsicherheit(data_alice_corr, 'A1B0')[0]}",
    f"nmax: {np.max(data_alice_corr['A1B0'])} pm {get_unsicherheit(data_alice_corr, 'A1B0')[1]}"
)
print(
    f"Polaristaionsgrad Koinzidenz A1B1:  {get_polaristaoinsgrad(data_alice_corr, 'A1B1')} +/- {get_unsicherheit_polarisationsgrad(data_alice_corr, 'A1B1')}",
    f"nmin: {np.min(data_alice_corr['A1B1'])} pm {get_unsicherheit(data_alice_corr, 'A1B1')[0]}",
    f"nmax: {np.max(data_alice_corr['A1B1'])} pm {get_unsicherheit(data_alice_corr, 'A1B1')[1]}",
)
print()
print()

plt.scatter(data_bob_corr['AngleBob(deg)'], data_bob_corr['A0B0'])
# plt.scatter(data_bob_corr['AngleBob(deg)'], data_bob_corr['A0B1'])
plt.title(r'\textbf{Coincidence count rate}' + '\n' + r' Alice HWP fixated at $0^\circ$')
plt.xlabel(r"Angle HWP $[^\circ]$", fontsize=FONTSIZE)
plt.ylabel(r"Counts $N$", fontsize=FONTSIZE)

# plt.show()
plt.savefig(
    KOINZIDENZ_PATH / 'KoinzidenztählratenAliceFixiert0_90_A0B0.jpg',
    dpi=900,
    bbox_inches="tight"
)
plt.close()
# plt.scatter(data_bob_corr['AngleBob(deg)'], data_bob_corr['A0B0'])
plt.scatter(data_bob_corr['AngleBob(deg)'], data_bob_corr['A0B1'])
plt.title(r'\textbf{Coincidence count rate}' + '\n' + r' Alice HWP fixated at $0^\circ$')
plt.xlabel(r"Angle HWP $[^\circ]$", fontsize=FONTSIZE)
plt.ylabel(r"Counts $N$", fontsize=FONTSIZE)
# plt.show()
plt.savefig(
    KOINZIDENZ_PATH / 'KoinzidenztählratenAliceFixiert0_90_A0B1.jpg',
    dpi=900,
    bbox_inches="tight"
)
plt.close()

plt.scatter(data_bob_corr['AngleBob(deg)'], data_bob_corr['A1B0'])
plt.title(r'\textbf{Coincidence count rate}' + '\n' + r' Alice HWP fixated at $0^\circ$')
plt.xlabel(r"Angle HWP $[^\circ]$", fontsize=FONTSIZE)
plt.ylabel(r"Counts $N$", fontsize=FONTSIZE)
# plt.show()
plt.savefig(
    KOINZIDENZ_PATH / 'KoinzidenztählratenAliceFixiert0_90_A1B0.jpg',
    dpi=900,
    bbox_inches="tight"
)
plt.close()

plt.scatter(data_bob_corr['AngleBob(deg)'], data_bob_corr['A1B1'])
plt.title(r'\textbf{Coincidence count rate}' + '\n' + r' Alice HWP fixated at $0^\circ$')
plt.xlabel(r"Angle HWP $[^\circ]$", fontsize=FONTSIZE)
plt.ylabel(r"Counts $N$", fontsize=FONTSIZE)
# plt.show()
plt.savefig(
    KOINZIDENZ_PATH / 'KoinzidenztählratenAliceFixiert0_90_A1B1.jpg',
    dpi=900,
    bbox_inches="tight"
)
plt.close()

plt.scatter(data_alice_corr['AngleAlice(deg)'], data_alice_corr['A0B0'])
plt.scatter(data_alice_corr['AngleAlice(deg)'], data_alice_corr['A1B0'])
plt.title(r'\textbf{Coincidence count rate}' + '\n' + r' Bob HWP fixated at $0^\circ$')
plt.xlabel(r"Angle HWP $[^\circ]$", fontsize=FONTSIZE)
plt.ylabel(r"Counts $N$", fontsize=FONTSIZE)
# plt.show()
plt.savefig(
    KOINZIDENZ_PATH / 'KoinzidenztählratenBobFixiert0.jpg',
    dpi=900,
    bbox_inches="tight"
)
plt.close()
