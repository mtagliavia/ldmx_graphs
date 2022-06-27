'''
This is an example that uses the functions in graph_functions.py to graph the
bin medians and quantiles of hit energy vs position
'''

from graph_functions.py import *

import numpy as np
import pickle

# unpack arrays from pickle files
dict_1GeV = pickle.load(open('./100k_1GeV.pkl', 'rb'))
xcoords1      = dict_1GeV['xcoords']
zcoords1      = dict_1GeV['zcoords']
rechitenergy1 = dict_1GeV['rechitenergy']

dict_2GeV = pickle.load(open('./100k_2GeV.pkl', 'rb'))
xcoords2      = dict_1GeV['xcoords']
zcoords2      = dict_1GeV['zcoords']
rechitenergy2 = dict_1GeV['rechitenergy']



# FIRST EXAMPLE: ENERGY BINNING ACROSS X-COORDINATES

# form the 2D arrays that correspond to coordinates of (xcoords, rechitenergy)
xenergy1 = np.zeros((xcoords1.size, 2), dtype='float')
xenergy1[:,0] = xcoords1
xenergy1[:,1] = rechitenergy1

xenergy2 = np.zeros((xcoords2.size, 2), dtype='float')
xenergy2[:,0] = xcoords2
xenergy2[:,1] = rechitenergy2


# use the first function (calc_bins) to find a consistent binning for both
# 1GeV and 2GeV data. We do this based on the x-coordinates because we want to
# know how the distribution of hit energies change based on position
bins = calc_bins(xenergy1[:,0], xenergy2[:,0], nbins=17)


# use this returned bins array as an input in the function that finds the
# median and desired quantiles of the positions and corresponding energies
# that fall into each bin
med1, qlow1, qhigh1 = ms_and_qs(xenergy1, bins)
med2, qlow2, qhigh2 = ms_and_qs(xenergy2, bins)


# put these returned 2D arrays into single 3D arrays for graphing purposes
meds = np.zeros((2, m1.shape[0], 2), dtype='float')
meds[0] = med1
meds[1] = med2

qlows = np.zeros((2, m1.shape[0], 2), dtype='float')
qlows[0] = qlow1
qlows[1] = qlow2

qhighs = np.zeros((2, m1.shape[0], 2), dtype='float')
qhighs[0] = qhigh1
qhighs[1] = qhigh2

energies = [1, 2]  # labels the 1GeV and 2GeV data in the graph


# now graph the data using the ms_qs_graph function
ms_qs_graph(meds, qlows, qhighs, 'x', energies, loc=9)
