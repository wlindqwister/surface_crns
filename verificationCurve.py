import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Data
dQdt_simulation = [4.90E-06, 3.15E-06, 2.06E-06, 1.37E-06, 9.24E-07, 7.18E-07, 5.95E-07, 7.79E-07, 7.92E-07, 7.88E-07, 7.41E-07, 
                   8.15E-07, 7.58E-07, 8.52E-07, 9.29E-07, 1.07E-06, 1.51E-06, 1.55E-06, 1.53E-06, 1.57E-06, 1.54E-06, 1.59E-06]
M0 = [0.968584073, 0.929314165, 0.874336294, 0.803650459, 0.717256661, 0.6151549, 0.497345175, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
      0.7, 0.7, 0.7975, 0.7979, 0.7983, 0.7987, 0.7991, 0.7995]
M1 = [62.8318, 94.2477, 125.6636, 157.0795, 188.4954, 219.9113, 251.3272, 50, 57.53278113, 63.80005894, 70.90416972, 78.53981634, 
      57.53278113, 86.57238995, 103.4656902, 121.1056027, 157.0795, 165.0795, 173.0795, 181.0795, 189.0795, 197.0795]
M2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -2, -3, -4]

# Normalize M1
min_M1 = min(M1)
max_M1 = max(M1)
M1_normalized = [(x - min_M1) / (max_M1 - min_M1) for x in M1]

# Functions
def exponentialFunc(inputs, Q, a, b, c):
    m0, m1, m2 = inputs
    return Q * np.exp(-a * m0 - b * m1 - c * m2)

def linearFunc(inputs, Q, a, b, c):
    m0, m1, m2 = inputs
    return Q + a * m0 + b * m1 + c * m2

def powerLawFunc(inputs, Q, a, b, c):
    m0, m1, m2 = inputs
    return Q * (m0 ** a) * (m1 ** b) * (m2 ** c)

def logFunc(inputs, Q, a, b, c):
    m0, m1, m2 = inputs
    return Q * np.log(m0) + a * np.log(m1) + b * np.log(m2) + c

# Prepare the input data for curve fitting
inputs = np.array([M0, M1_normalized, M2])

# Perform exponential curve fitting
popt_exp, pcov = curve_fit(exponentialFunc, inputs, dQdt_simulation)

# Extract the fitted parameters
Q_exp, a_exp, b_exp, c_exp = popt_exp

# Print the fitted parameters
print(f"Exponential fitted parameters: Q = {Q_exp}, a = {a_exp}, b = {b_exp}, c = {c_exp}")

# Calculate the fitted values
fitted_values_exp = exponentialFunc(inputs, Q_exp, a_exp, b_exp, c_exp)

# Perform exponential curve fitting
popt_lin, pcov = curve_fit(linearFunc, inputs, dQdt_simulation)

# Extract the fitted parameters
Q_lin, a_lin, b_lin, c_lin = popt_lin

# Print the fitted parameters
print(f"Linear fitted parameters: Q = {Q_lin}, a = {a_lin}, b = {b_lin}, c = {c_lin}")

# Calculate the fitted values
fitted_values_lin = linearFunc(inputs, Q_lin, a_lin, b_lin, c_lin)

# Perform power law curve fitting
popt_log, pcov = curve_fit(logFunc, inputs, dQdt_simulation)

# Extract the fitted parameters
Q_log, a_log, b_log, c_log = popt_log

# Print the fitted parameters
print(f"Log fitted parameters: Q = {Q_log}, a = {a_log}, b = {b_log}, c = {c_log}")

# Calculate the fitted values
fitted_values_log = logFunc(inputs, Q_log, a_log, b_log, c_log)

# Filter out NaN values for the logarithmic fit
mask = np.isfinite(fitted_values_log)
filtered_fitted_values_log = np.array(fitted_values_log)[mask]
filtered_dQdt_simulation = np.array(dQdt_simulation)[mask]

# Calculate R² values
r2_exp = r2_score(dQdt_simulation, fitted_values_exp)
r2_lin = r2_score(dQdt_simulation, fitted_values_lin)
r2_log = r2_score(filtered_dQdt_simulation, filtered_fitted_values_log)

# Print R² values
print(f"R² (Exponential): {r2_exp}")
print(f"R² (Linear): {r2_lin}")
print(f"R² (Logarithmic): {r2_log}")

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)

# Plot the original data and the fitted curves
# axs[0, 0].scatter(range(len(dQdt_simulation)), dQdt_simulation, label='Simulation Data')
# axs[0, 0].plot(range(len(dQdt_simulation)), fitted_values_exp, label='Fitted Exponential Curve', color='red', linestyle='dashed', alpha=0.5)
# axs[0, 0].plot(range(len(dQdt_simulation)), fitted_values_lin, label='Fitted Linear Curve', color='green', linestyle='dashed', alpha=0.5)
# axs[0, 0].legend()
# axs[0, 0].set_xlabel('Index')
# axs[0, 0].set_ylabel('$\\frac{dQ}{dt_{max}}$')
# axs[0, 0].set_title('Curve Fitting for Exponential and Linear Functions')

# Parity plot for both fits
axs[0, 0].scatter(dQdt_simulation, fitted_values_exp, label='Exponential Fit', color='blue', alpha=0.5)
axs[0, 0].scatter(dQdt_simulation, fitted_values_lin, label='Linear Fit', color='orange', alpha=0.5)
axs[0, 0].scatter(dQdt_simulation, fitted_values_log, label='Power Fit', color='slategray', alpha=0.5)
min_val = min(dQdt_simulation)
max_val = max(dQdt_simulation)
axs[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--')  # Line with slope of 1
axs[0, 0].legend()
axs[0, 0].set_xlabel('Simulation Data')
axs[0, 0].set_ylabel('Fitted Data')
axs[0, 0].set_title('Parity Plot for All Fits')
axs[0, 0].set_xscale('log')
axs[0, 0].set_yscale('log')

# Parity plot for exponential fit
axs[0, 1].scatter(dQdt_simulation, fitted_values_exp, label='Exponential Fit', color='blue', alpha=0.5)
axs[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--')  # Line with slope of 1
axs[0, 1].set_xlabel('Simulation Data')
axs[0, 1].set_ylabel('Fitted Data')
axs[0, 1].set_title('Parity Plot, Exponential Fit')
axs[0, 1].text(0.05, 0.85, f'R²: {r2_exp:.2f}', transform=axs[0, 1].transAxes, verticalalignment='top', fontweight='bold')

# Parity plot for linear fit
axs[1, 0].scatter(dQdt_simulation, fitted_values_lin, label='Linear Fit', color='orange', alpha=0.5)
axs[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--')  # Line with slope of 1
axs[1, 0].set_xlabel('Simulation Data')
axs[1, 0].set_ylabel('Fitted Data')
axs[1, 0].set_title('Parity Plot, Linear Fit')
axs[1, 0].text(0.05, 0.85, f'R²: {r2_lin:.2f}', transform=axs[1, 0].transAxes, verticalalignment='top', fontweight='bold')

# Parity plot for linear fit
axs[1, 1].scatter(dQdt_simulation, fitted_values_log, label='Power Fit', color='slategray', alpha=0.5)
axs[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--')  # Line with slope of 1
axs[1, 1].set_xlabel('Simulation Data')
axs[1, 1].set_ylabel('Fitted Data')
axs[1, 1].set_title('Parity Plot, Log Fit')
axs[1, 1].set_xscale('log')
axs[1, 1].set_yscale('log')
axs[1, 1].text(0.05, 0.85, f'R²: {r2_log:.2f}', transform=axs[1, 1].transAxes, verticalalignment='top', fontweight='bold')

plt.tight_layout()
plt.show()

# # Plot the original data and the fitted curve
# plt.figure()
# plt.scatter(range(len(dQdt_simulation)), dQdt_simulation, label='Simulation Data')
# plt.plot(range(len(dQdt_simulation)), fitted_values_exp, 
#       label='Fitted Exponential Curve', color='red', linestyle='dashed', alpha = 0.5)
# plt.plot(range(len(dQdt_simulation)), fitted_values_lin, 
#       label='Fitted Linear Curve', color='green', linestyle='dashed', alpha = 0.5)
# plt.legend()
# plt.xlabel('Index')
# plt.ylabel('dQ/dt')
# plt.title('Curve Fitting for Exponential Function')

# # Parity plot fit
# plt.figure()
# plt.scatter(dQdt_simulation, fitted_values_exp, 
#       label='Exponential Fit', color = 'red', alpha = 0.5)
# plt.scatter(dQdt_simulation, fitted_values_lin, 
#       label='Linear Fit', color = 'green', alpha = 0.5)
# min_val = min(dQdt_simulation)
# max_val = max(dQdt_simulation)
# plt.plot([min_val, max_val], [min_val, max_val], 'k--')  # Line with slope of 1
# plt.legend()
# plt.xlabel('Simulation Data')
# plt.ylabel('Fitted Data')
# plt.title('Parity Plot')

# # Parity plot exponential fit
# plt.figure()
# plt.scatter(dQdt_simulation, fitted_values_exp, 
#       label='Exponential Fit', color = 'red', alpha = 0.5)
# min_val = min(dQdt_simulation)
# max_val = max(dQdt_simulation)
# plt.plot([min_val, max_val], [min_val, max_val], 'k--')  # Line with slope of 1
# plt.xlabel('Simulation Data')
# plt.ylabel('Fitted Data')
# plt.title('Parity Plot, Exponential Fit')

# # Parity plot linear fit
# plt.figure()
# plt.scatter(dQdt_simulation, fitted_values_lin, 
#       label='Linear Fit', color = 'green', alpha = 0.5)
# min_val = min(dQdt_simulation)
# max_val = max(dQdt_simulation)
# plt.plot([min_val, max_val], [min_val, max_val], 'k--')  # Line with slope of 1
# plt.xlabel('Simulation Data')
# plt.ylabel('Fitted Data')
# plt.title('Parity Plot, Linear Fit')

# plt.show()