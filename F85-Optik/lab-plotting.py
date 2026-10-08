import matplotlib.pyplot as plt

high_volt = [0, 0.18, 0.35, 0.52, 0.71, 0.88, 1.06, 1.24, 1.41, 1.59, 1.78]

photo_volt_plus = [1.14, 1.04, 1.660, 1.92, 1.30, 1.02, 1.520, 1.920, 1.420, 1.00, 1.520]

photo_volt_minus = [1.040, 1.080, 1.640, 1.760, 1.240, 1.040,1.500, 1.720, 1.340, 1.100, 1.480]

print(len(high_volt), len(photo_volt_minus), len(photo_volt_plus))


plot = plt.scatter(high_volt, photo_volt_minus)
plt.show()

plot_2 = plt.scatter(high_volt, photo_volt_plus)
plt.show()
