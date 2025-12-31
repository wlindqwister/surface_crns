import numpy as np
import matplotlib.pyplot as plt
import porespy as ps
import time
import pickle
import pandas as pd
import csv
import os
import math
from os import listdir
from numpy.polynomial.polynomial import polyfit
from whittaker_eilers import WhittakerSmoother
from math import factorial
from tqdm import tqdm
from PIL import Image, ImageDraw
from skimage.measure import perimeter, euler_number
from scipy.optimize import curve_fit
from surface_crns import SurfaceCRNQueueSimulator
from surface_crns.models.grids import SquareGrid
from surface_crns.readers.manifest_readers import read_manifest
from surface_crns.options.option_processor import SurfaceCRNOptionParser
from surface_crns.simulators.queue_simulator import QueueSimulator

def reactionquotient(conc):
    #return (np.array(conc['R'])*(np.array(conc['R']))) / ((np.array(conc['Q']) * np.array(conc['A'])))
    return np.array(conc['R']) / ((np.array(conc['Q']) * np.array(conc['A'])))

def reactionquotient_whit(conc, slice_factor, plots = True):
    whittaker_smoother = WhittakerSmoother(lmbda=1000, order=2, data_length=len(conc['Q'][::slice_factor]))
    print('Smoothing Data...')
    print('Q')
    Qhat = np.array(whittaker_smoother.smooth(conc['Q'][::slice_factor]))
    print('A')
    Ahat = np.array(whittaker_smoother.smooth(conc['A'][::slice_factor]))
    print('R')
    Rhat = np.array(whittaker_smoother.smooth(conc['R'][::slice_factor]))
    print('Data Smoothing Complete')
    
    if plots:
        plt.figure(figsize=(8,6))
        for species in species_tracked:
            plt.plot(conc['time'], conc[species], label = species)
        plt.plot(conc['time'][::slice_factor], Qhat)
        plt.plot(conc['time'][::slice_factor], Ahat)
        plt.plot(conc['time'][::slice_factor], Rhat)
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Packet Count (#)")
        plt.title("Evolution of QAR Species")
        plt.show()
    
    
    #return (Rhat * Rhat) / (Qhat * Ahat)
    return Rhat / (Qhat * Ahat)

def find_max_value(dataset):
    max_value = float('-inf')
    max_index = None

    for i, data_point in enumerate(dataset):
        if data_point > max_value:
            max_value = data_point
            max_index = i

    return max_value, max_index

def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in sorted(filenames) if filename.endswith( suffix ) ]

MAIN_DIR = os.getcwd()
DATA_DIR = MAIN_DIR + '/data/'

species_tracked = ['Q', 'A', 'R']

short_step = 1000

slicer = 1

uc_names = find_csv_filenames(DATA_DIR)

uc_data = {}

for file in tqdm(uc_names):
    reader = csv.DictReader(open(DATA_DIR + file))

    data = {}

    for row in reader:
        for column, value in row.items(): # consider .iteritems() for Python 2
             data.setdefault(column, []).append(float(value))

    rxn_quotient = reactionquotient_whit(data, slicer, plots = False)

    uc_data[file] = rxn_quotient
    uc_data[file + "_time"] = data['time']

    del rxn_quotient

# Shorten the analysis window
uc_data_shortened = {}

for file in tqdm(uc_names):
    indices = [i for i, x in enumerate(uc_data[file + "_time"]) if x < short_step]
    uc_data_shortened[file] = [uc_data[file][i] for i in indices]
    uc_data_shortened[file + "_time"] = [uc_data[file + "_time"][i] for i in indices]

radii = ['10', '15', '20', '25', '30', '35', '40', '45', '50']

#plt.figure(figsize=(16,10))
for i, dataset in enumerate(uc_names):
    #plt.plot(uc_data[dataset + "_time"][::slicer], uc_data[dataset], label = radii[i])
    plt.plot(uc_data_shortened[dataset + "_time"][::slicer], uc_data_shortened[dataset], label = radii[i])
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Q")
plt.title("Evolution of Q, Unit Cell Test")

plt.show()

# Calculate derivatives
# uc_derivs = {}
# for name in tqdm(uc_names):
#     grad = np.gradient(uc_data[name], uc_data[name  + "_time"][::slicer])
#     w_s = WhittakerSmoother(lmbda=50000, order=1, data_length=len(grad),
#                         x_input = uc_data[name + "_time"][::slicer])
#     derivhat = w_s.smooth(grad)
#     #uc_derivs[name] = grad
#     uc_derivs[name] = derivhat

# for i, dataset in enumerate(uc_names):
#     plt.plot(uc_data[dataset + "_time"][::slicer], uc_derivs[dataset], label = radii[i])
# plt.legend()
# plt.xlabel("Time (s)")
# plt.ylabel("dQ/dt")
# plt.title("Evolution of Q derivative, Unit Cell Test")

uc_derivs_shortened = {}
for name in tqdm(uc_names):
    grad = np.gradient(uc_data_shortened[name], uc_data_shortened[name  + "_time"][::slicer])
    w_s = WhittakerSmoother(lmbda=50000, order=1, data_length=len(grad),
                        x_input = uc_data_shortened[name + "_time"][::slicer])
    derivhat = w_s.smooth(grad)
    #uc_derivs[name] = grad
    uc_derivs_shortened[name] = derivhat

for i, dataset in enumerate(uc_names):
    plt.plot(uc_data_shortened[dataset + "_time"][::slicer], uc_derivs_shortened[dataset], label = radii[i])
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("dQ/dt")
plt.title("Evolution of Q derivative, Unit Cell Test")

plt.show()

firstderivtime = []
maxfirstderiv = []
for name in uc_names:
    max_value, max_index = find_max_value(uc_derivs_shortened[name])
    firstderivtime.append(uc_data_shortened[name  + "_time"][::slicer][max_index])
    maxfirstderiv.append(max_value)

print(maxfirstderiv)

popt, pcov = curve_fit(exponential_func, radii, maxfirstderiv)
print(popt)

print(firstderivtime)

plt.plot(list(map(int, radii)), maxfirstderiv, 'ko')
plt.plot(list(map(int, radii)), maxfirstderiv, 'r--')
#plt.plot(list(map(int, radii)), exponential_func(np.array(list(map(int, radii))), 0.00002, 0.1, 0), 'r-')
plt.xlabel('Radius')
plt.ylabel('Maximum $\\frac{dQ}{dt}$')
plt.show()

plt.plot(list(map(int, radii)), firstderivtime, 'ko')
plt.xlabel('Radius')
plt.ylabel('Time of max 1st deriv')
plt.show()