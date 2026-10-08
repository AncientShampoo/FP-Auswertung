import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from odrpack import odr_fit
import pathlib
import os

BASE_DIR = pathlib.Path(__file__).parent
IMAGES = BASE_DIR / "Images" / "partZwei"
print(IMAGES)

# 2.21
## Measurements Data
dc_offset_volt = [
    0.00,
    0.30,
    0.60,
    0.90,
    1.20,
    1.50,
    1.80,
    2.10,
    2.40,
    2.70,
    3.00,
    3.30,
    3.60,
    3.90,
    4.20,
    4.50,
    4.80,
    5.10,
    5.40,
    5.70,
    6.00,
]
dc_offset_V_err = np.array([
    0.04,
    0.03,
    0.04,
    0.03,
    0.04,
    0.03,
    0.04,
    0.04,
    0.04,
    0.04,
    0.04,
    0.04,
    0.04,
    0.06,
    0.06,
    0.06,
    0.06,
    0.10,
    0.20,
    0.30,
    0.30,
])
output_Voltage = [
    0.00,
    0.07,
    0.16,
    0.26,
    0.35,
    0.44,
    0.51,
    0.61,
    0.71,
    0.79,
    0.87,
    0.96,
    1.06,
    1.15,
    1.24,
    1.33,
    1.40,
    1.50,
    1.58,
    1.68,
    1.77,
]
output_Voltage_err = np.ones_like(output_Voltage) * 0.01
# Sanity checking
# print(len(dc_offset_volt), len(dc_offset_V_err), len(output_Voltage))


# Evaluation
# # def linear function for curve fitting
def func_linear(x, beta: np.ndarray):
    a, b = beta
    return a * x + b

beta0 = [1.0, 1.0]
weights_x = 1/dc_offset_V_err**2
weights_y = 1/output_Voltage_err**2

sol = odr_fit(
    func_linear,
    dc_offset_volt,
    output_Voltage,
    beta0=beta0,
    weight_x=weights_x,
    weight_y=weights_y
)
# p_opt, p_cov = curve_fit(func_linear, dc_offset_volt, output_Voltage)

# print(p_opt)
# print(p_cov)

x_vals_fitt = np.linspace(0, 6, 1000)
y_vals_fitt = func_linear(x_vals_fitt, sol.beta)
p_vals_err = np.sqrt(np.diag(sol.cov_beta))
print(p_vals_err)

# plt.scatter(dc_offset_volt, output_Voltage)
plt.errorbar(dc_offset_volt, output_Voltage, output_Voltage_err, dc_offset_V_err, fmt=".")
plt.plot(
    x_vals_fitt,
    y_vals_fitt,
    label=f"linear fitt a={np.round(sol.beta[0], 3)} +/- {np.round(p_vals_err[0], 3)}"
    + f", b={np.round(sol.beta[1], 3)} +/- {np.round(p_vals_err[1], 3)}",
)
plt.legend()
plt.savefig(IMAGES/"part_2.2.1_linear_fitt_output_input.jpg", dpi=900)
# plt.show()
plt.close()

##############################################
## -------------- Part 2.2.3 -------------- ##
##############################################
print("\nNow at part 2: \n")
## Measurements Data
output_Voltage_2_plus = np.array(
    [
        0.0,
        0.18,
        0.35,
        0.53,
        0.71,
        0.88,
        1.06,
        1.24,
        1.41,
        1.58,
        1.78,
        0.20,
        0.23,
        0.30,
        0.55,
        0.59,
        0.65,
    ]
)
output_Voltage_2_minus = np.array(
    [
        0.00,
        0.18,
        0.35,
        0.52,
        0.70,
        0.89,
        1.07,
        1.24,
        1.41,
        1.59,
        1.77,
        0.20,
        0.25,
        0.30,
        0.55,
        0.59,
        0.65,
    ]
)

photo_diode_plus = np.array(
    [
        1.140,
        1.040,
        1.660,
        1.920,
        1.300,
        1.020,
        1.520,
        1.920,
        1.420,
        1.000,
        1.520,
        0.980,
        0.960,
        1.000,
        1.820,
        1.840,
        1.800,
    ]
)
photo_diode_minus = np.array(
    [
        1.040,
        1.080,
        1.640,
        1.760,
        1.240,
        1.040,
        1.500,
        1.720,
        1.340,
        1.100,
        1.480,
        1.080,
        1.120,
        1.200,
        1.800,
        1.760,
        1.660,
    ]
)

ph_diode_plus_err = np.zeros(len(photo_diode_plus)) + 0.020
ph_diode_minus_err = np.zeros(len(photo_diode_minus)) + 0.020

# sorting arrays in x values for easier handling
idx_sorted_plus = np.argsort(output_Voltage_2_plus)
output_Voltage_2_plus_sort = output_Voltage_2_plus[idx_sorted_plus]
photo_diode_plus_sort = photo_diode_plus[idx_sorted_plus]

idx_sorted_minus = np.argsort(output_Voltage_2_minus)
output_Voltage_2_minus_sort = output_Voltage_2_minus[idx_sorted_minus]
photo_diode_minus_sort = photo_diode_minus[idx_sorted_minus]

# print(len(output_Voltage_2_plus_sort), len(output_Voltage_2_minus), len(photo_diode_plus), len(photo_diode_minus))


# def function for curve fit (sin)
# def func_sin_fitt(x, a, b, c, d):
#     return a * np.sin(b * (x + c)) + d
def func_sin_fitt(x, beta0):
    a, b, c, d = beta0
    return a * np.sin(b * (x + c)) + d


####################################################
# ----------------- +45 Degrees ------------------ #
####################################################
# determine starting values for optimization (plus angle)
max_ph_diod_plus = np.max(photo_diode_plus_sort[0:11])
min_ph_diod_plus = np.min(photo_diode_plus_sort[0:11])
idx_max_ph_plus = np.argmax(photo_diode_plus_sort[0:11])
idx_min_ph_plus = np.argmin(photo_diode_plus_sort[0:11])

start_vals_optim_plus = [
    (max_ph_diod_plus - min_ph_diod_plus) / 2,
    np.pi
    / (
        output_Voltage_2_plus_sort[idx_max_ph_plus]
        - output_Voltage_2_plus_sort[idx_min_ph_plus]
    ),
    (
        output_Voltage_2_plus_sort[idx_max_ph_plus]
        + output_Voltage_2_plus_sort[idx_min_ph_plus]
    )
    / 2,
    (max_ph_diod_plus + min_ph_diod_plus) / 2,
]

print(start_vals_optim_plus)
# p_opt, p_cov = curve_fit(
#     func_sin_fitt,
#     output_Voltage_2_plus_sort,
#     photo_diode_plus_sort,
#     p0=start_vals_optim_plus,
# )
sol = odr_fit(
    func_sin_fitt,
    output_Voltage_2_plus_sort,
    photo_diode_plus_sort,
    beta0=start_vals_optim_plus,
    weight_x = (1/0.01**2),
    weight_y = (1/0.02**2),
    maxit=2000,
    scale_beta=[0.1, 0.001, 0.10, 0.10],
    # report='long'
)

print(sol.beta)
x_vals_fitt = np.linspace(0, 1.8, 1000)
y_vals_fitt = func_sin_fitt(x_vals_fitt, sol.beta)
# y_vals_fitt = func_sin_fitt(x_vals_fitt, 0.72 ,8.5, -0.35, 1.36)
p_vals_err = np.sqrt(np.diag(sol.cov_beta))
print(p_vals_err)

# # Plotting +45 Degrees
plt.scatter(output_Voltage_2_plus_sort, photo_diode_plus_sort)
plt.plot(x_vals_fitt, y_vals_fitt)
# plt.scatter(output_Voltage_2_minus, photo_diode_minus)
plt.savefig(
    IMAGES / "part_2.2.3_fitt_sinus_plus.jpg",
    dpi=900
)
# plt.show()
plt.close()

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# Optimization without additional measurement points#
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# # determine starting values for optimization (plus angle)
max_ph_diod_plus = np.max(photo_diode_plus[0:5])
min_ph_diod_plus = np.min(photo_diode_plus[0:5])
idx_max_ph_plus = np.argmax(photo_diode_plus[0:5])
idx_min_ph_plus = np.argmin(photo_diode_plus[0:5])

start_vals_optim_plus = [
    (max_ph_diod_plus - min_ph_diod_plus) / 2,
    np.pi
    / (output_Voltage_2_plus[idx_max_ph_plus] - output_Voltage_2_plus[idx_min_ph_plus]),
    (output_Voltage_2_plus[idx_max_ph_plus] + output_Voltage_2_plus[idx_min_ph_plus])
    / 2,
    (max_ph_diod_plus + min_ph_diod_plus) / 2,
]

# p_opt_without, p_cov_without = curve_fit(
#     func_sin_fitt,
#     output_Voltage_2_plus[:-6],
#     photo_diode_plus[:-6],
#     p0=start_vals_optim_plus,
# )

sol = odr_fit(
    func_sin_fitt,
    output_Voltage_2_plus[:-6],
    photo_diode_plus[:-6],
    beta0=start_vals_optim_plus,
    weight_x = (1/0.01**2),
    weight_y = (1/0.02**2),
    maxit=2000,
    scale_beta=[0.1, 0.001, 0.10, 0.10],
)
x_vals_fitt_without = np.linspace(0, 1.8, 1000)
y_vals_fitt_without = func_sin_fitt(x_vals_fitt_without, sol.beta)
# y_vals_fitt_without = func_sin_fitt(x_vals_fitt_without, 0.72 ,8.5, -0.35, 1.36)
p_vals_err = np.sqrt(np.diag(sol.cov_beta))
print("\nValues found without extra points: ")
print(sol.beta)
print(p_vals_err, "\n")

plt.scatter(output_Voltage_2_plus[:-6], photo_diode_plus[:-6])
plt.plot(x_vals_fitt_without, y_vals_fitt_without)
# plt.scatter(output_Voltage_2_minus, photo_diode_minus)
plt.savefig(IMAGES / "part_2.2.3_fitt_sinus_plus_without_add_measurements.jpg", dpi=900)
# plt.show()
plt.close()

plt.scatter(output_Voltage_2_plus, photo_diode_plus)
plt.plot(x_vals_fitt_without, y_vals_fitt_without)
plt.plot(x_vals_fitt, y_vals_fitt)
# plt.scatter(output_Voltage_2_minus, photo_diode_minus)
plt.savefig(
    IMAGES / "part_2.2.3_fitt_sinus_plus_without_add_measurements_comparison.jpg", dpi=900
)
# plt.show()
plt.close()


#####################################################
## ---------------- -45 Degrees ------------------ ##
#####################################################
# determine starting values for optimization (minus angle)

max_ph_diod_minus = np.max(photo_diode_minus_sort[0:11])
min_ph_diod_minus = np.min(photo_diode_minus_sort[0:11])
idx_max_ph_minus = np.argmax(photo_diode_minus_sort[0:11])
idx_min_ph_minus = np.argmin(photo_diode_minus_sort[0:11])

start_vals_optim_minus = [
    (max_ph_diod_minus - min_ph_diod_minus) / 2,
    np.pi
    / (
        output_Voltage_2_minus_sort[idx_max_ph_minus]
        - output_Voltage_2_minus_sort[idx_min_ph_minus]
    ),
    (
        output_Voltage_2_minus_sort[idx_max_ph_minus]
        + output_Voltage_2_minus_sort[idx_min_ph_minus]
    )
    / 2,
    (max_ph_diod_minus + min_ph_diod_minus) / 2,
]

print(start_vals_optim_minus)
# p_opt, p_cov = curve_fit(
#     func_sin_fitt,
#     output_Voltage_2_minus_sort,
#     photo_diode_minus_sort,
#     p0=start_vals_optim_minus,
# )

sol = odr_fit(
    func_sin_fitt,
    output_Voltage_2_minus_sort,
    photo_diode_minus_sort,
    beta0=start_vals_optim_minus,
    weight_x = (1/0.01**2),
    weight_y = (1/0.02**2),
    maxit=2000,
    scale_beta=[0.1, 0.001, 0.10, 0.10]
)

print(sol.beta)
x_vals_fitt = np.linspace(0, 1.8, 1000)
y_vals_fitt = func_sin_fitt(x_vals_fitt, sol.beta)
# y_vals_fitt = func_sin_fitt(x_vals_fitt, 0.72 ,8.5, -0.35, 1.36)
p_vals_err = np.sqrt(np.diag(sol.cov_beta))
print(p_vals_err)


# # Plotting -45 Degrees
plt.scatter(output_Voltage_2_minus_sort, photo_diode_minus_sort)
plt.plot(x_vals_fitt, y_vals_fitt)
# plt.scatter(output_Voltage_2_minus, photo_diode_minus)
plt.savefig(
    IMAGES / "part_2.2.3_fitt_sinus_minus.jpg",
    dpi=900
)
# plt.show()
plt.close()

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# Optimization without additional measurement points#
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# # determine starting values for optimization (plus angle)
max_ph_diod_minus = np.max(photo_diode_minus[0:5])
min_ph_diod_minus = np.min(photo_diode_minus[0:5])
idx_max_ph_minus = np.argmax(photo_diode_minus[0:5])
idx_min_ph_minus = np.argmin(photo_diode_minus[0:5])

start_vals_optim_minus = [
    (max_ph_diod_minus - min_ph_diod_minus) / 2,
    np.pi
    / (
        output_Voltage_2_minus[idx_max_ph_minus]
        - output_Voltage_2_minus[idx_min_ph_minus]
    ),
    (
        output_Voltage_2_minus[idx_max_ph_minus]
        + output_Voltage_2_minus[idx_min_ph_minus]
    )
    / 2,
    (max_ph_diod_minus + min_ph_diod_minus) / 2,
]



# p_opt_without, p_cov_without = curve_fit(
#     func_sin_fitt,
#     output_Voltage_2_minus[:-6],
#     photo_diode_minus[:-6],
#     p0=start_vals_optim_minus,
# )

sol = odr_fit(
    func_sin_fitt,
    output_Voltage_2_minus[:-6],
    photo_diode_minus[:-6],
    beta0=start_vals_optim_plus,
    weight_x = (1/0.01**2),
    weight_y = (1/0.02**2),
    maxit=2000,
    scale_beta=[0.1, 0.001, 0.10, 0.10]
)

x_vals_fitt_without = np.linspace(0, 1.8, 1000)
y_vals_fitt_without = func_sin_fitt(x_vals_fitt_without, sol.beta)
# y_vals_fitt_without = func_sin_fitt(x_vals_fitt_without, 0.72 ,8.5, -0.35, 1.36)
p_vals_err = np.sqrt(np.diag(sol.cov_beta))
print("\nValues found without extra points: ")
print(sol.beta)
print(p_vals_err, "\n")

plt.scatter(output_Voltage_2_minus[:-6], photo_diode_minus[:-6])
plt.plot(x_vals_fitt_without, y_vals_fitt_without)
# plt.scatter(output_Voltage_2_minus, photo_diode_minus)
plt.savefig(
    IMAGES / "part_2.2.3_fitt_sinus_minus_without_add_measurements.jpg",
    dpi=900
)
# plt.show()
plt.close()

plt.scatter(output_Voltage_2_minus, photo_diode_minus)
plt.plot(x_vals_fitt_without, y_vals_fitt_without)
plt.plot(x_vals_fitt, y_vals_fitt)
# plt.scatter(output_Voltage_2_minus, photo_diode_minus)
plt.savefig(
    IMAGES / "part_2.2.3_fitt_sinus_minus_without_add_measurements_comparison.jpg", dpi=900
)
# plt.show()
plt.close()
