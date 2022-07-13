'''
This is where my graphing functions will go. I will import these functions into
the documents I use to analyze the simulation data. This file includes the
functions in graph_functions.py.
'''

import copy
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


def hist1(x, title, label, size=(10,10), nbins=25, xlabel=None, ylabel=None,
          xlim=(None,None), yscale='linear', loc='best', fs=20):
    '''
    This function produces a 1D histogram of the input data.
    
    Inputs:
        - x (1D array) : numpy array to be histogrammed
        - title (str)  : title of graph
        - label (str)  : description of particles fired; appears in graph
                         legend
    Kwargs:
        - size (tuple) : figure size of graph produced
        - nbins (int)  : number of bins in the histogram
        - xlabel (str) : x-axis label
        - ylabel (str) : y-axis label
        - xlim (tuple) : range of x-values to display in the graph
        - yscale (str) : how to scale the values in each bin
        - loc (str/int): location of the graph legend; 1 is upper right, 5 is
                         middle right, 9 is top middle
        - fs (int)     : font size of legend
        
    Returns:
        - an mplhep histogram plot
    '''
    plt.figure(figsize=size)
    h, bins = np.histogram(x, bins=nbins)
#    xxx = plt.axes(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.title(title, size=20)
    plt.xlabel(xlabel, size=18)
    plt.ylabel(ylabel, size=18)
    hep.histplot(h, bins, label=label)
    plt.xlim(xlim)
    plt.yscale(yscale)
    plt.legend(loc=loc, fontsize=fs)
    plt.show()
    
    
    
def scatter1(x, y, title, xlabel, ylabel, label, size=(10,10), s=1,
             xlim=(None,None), ylim=(None,None), yscale='linear', loc='best',
             fs=18):
    '''
    This function produces a 2D scatterplot of the input data.
    
    Inputs:
        - x (1D array) : numpy array that contains the horizontal coordinates
        - y (1D array) : numpy array that contains the vertical coordinates
        - title (str)  : title of graph
        - xlabel (str) : x-axis label
        - ylabel (str) : y-axis label
        - label (str)  : description of particles fired; appears in graph
                         legend
    Kwargs:
        - size (tuple) : figure size of graph produced
        - s (float)    : size of points graphed
        - xlim (tuple) : range of x-values to display in the graph
        - ylim (tuple) : range of y-values to display in the graph
        - yscale (str) : how to scale the values in each bin
        - loc (str/int): location of the graph legend; 1 is upper right, 5 is
                         middle right, 9 is top middle
        - fs (int)     : font size of legend
        
    Returns:
        - a matplotlib.pyplot scatterplot
    '''
    plt.figure(figsize=size)
    plt.scatter(x, y, s=s, label=label)
    plt.title(title, size=20)
    plt.xlabel(xlabel, size=18)
    plt.ylabel(ylabel, size=18)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.yscale(yscale)
    plt.legend(loc=loc, fontsize=fs)
    plt.show()



def scatter_multi(x, y, title, xlabel, ylabel, label, size=(10,10), s=1,
                  xlim=(None,None), ylim=(None,None), yscale='linear',
                  loc='best', fs=18, colors=None):
    '''
    This function produces a 2D scatterplot of input data from multiple
    datasets.
    
    Inputs:
        - x (list)     : contains 1D numpy arrays that contain the horizontal
                         coordinates
        - y (list)     : contains 1D numpy arrays that contain the vertical
                         coordinates
        - title (str)  : title of graph
        - xlabel (str) : x-axis label
        - ylabel (str) : y-axis label
        - label (list) : description of particles fired; appears in graph
                         legend
    Kwargs:
        - size (tuple) : figure size of graph produced
        - s (float)    : size of points graphed
        - xlim (tuple) : range of x-values to display in the graph
        - ylim (tuple) : range of y-values to display in the graph
        - yscale (str) : how to scale the values in each bin
        - loc (str/int): location of the graph legend; 1 is upper right, 5 is
                         middle right, 9 is top middle
        - fs (int)     : font size of legend
        - colors (list): colors of datasets in plot
        
    Returns:
        - a matplotlib.pyplot scatterplot
    '''
    if colors != None:
        cs = colors
    else:
        cs = plt.cm.Set1(np.linspace(0, 1, len(x)))
        
    plt.figure(figsize=size)
    for i in range(len(x)):
        plt.scatter(x[i], y[i], s=s, label=label[i], color=cs[i])
    plt.title(title, size=20)
    plt.xlabel(xlabel, size=18)
    plt.ylabel(ylabel, size=18)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.yscale(yscale)
    plt.legend(loc=loc, fontsize=fs)
    plt.show()

    
    
def hists(xs, title, labels, size=(10,10), nbins=25, alpha=0.6, colors=None,
          edge=True, xlabel=None, ylabel=None, xlim=(None,None),
          yscale='linear', loc='best', fs=20, cumulative=0):
    '''
    This function graphs many histograms on the same plot.
    
    Inputs:
        - xs (2D array)      : 2D numpy array where each element is one
                               dataset to be histogrammed
        - title (str)        : title of graph
        - label (list)       : description of particles fired; appears in
                               graph legend
    Kwargs:
        - size (tuple)       : figure size of graph produced
        - nbins (int/array)  : number of bins in each histogram; can
                               'customize' for each individual histogram; can
                               use output of calc_bins function
        - alpha (float/array): opacity of histogram bars; can 'customize' for
                               each individual histogram
        - colors (array)     : color of each histogram; can 'customize' for
                               each individual histogram
        - edge (bool)        : whether to graph edges or fill histogram bins
        - xlabel (str)       : x-axis label
        - ylabel (str)       : y-axis label
        - xlim (tuple)       : range of x-values to display in the graph
        - yscale (str)       : how to scale the values in each bin
        - loc (str/int)      : location of the graph legend; 1 is upper right,
                               5 is middle right, 9 is top middle
        - fs (int)           : font size of legend
        - cumulative (int)   : set equal to 1/-1 to graph integral/1-integral
                               of graph
        
    Returns:
        - a matplotlib.pyplot graph of many histograms
    '''
    if type(nbins)==int:
        nb = np.ones(len(xs), dtype='int')*nbins
    elif len(nbins)!=len(xs):
        nb = np.zeros((len(xs), nbins.size), dtype='float')
        for i in range(nb.shape[0]):
            nb[i] = nbins
    else:
        nb = nbins
        
    if type(alpha)==float:
        a = np.ones(len(xs))*alpha
    else:
        a = alpha
        
    if colors != None:
        cs = colors
    else:
        cs = plt.cm.Set1(np.linspace(0, 1, len(xs)))
    
    plt.figure(figsize=size)
    
    if edge==True:
        hmin = np.zeros((len(xs), nb[0].size), dtype='float')
        hmax = np.zeros((len(xs), nb[0].size), dtype='float')
        for i in range(len(xs)):
            h = plt.hist(xs[i], bins=nb[i], label=labels[i], alpha=1,
                         edgecolor=cs[i], linewidth=2, fill=False,
                         cumulative=cumulative)
            plt.vlines(nb[i], 0, h[0].max(), colors='white', alpha=1, linewidth=2)
            h_list_left = list(h[0])
            h_list_left.append(0)
            h_list_right = list(h[0][::-1])
            h_list_right.append(0)
            hl = np.array(h_list_left)
            hr = np.array(h_list_right)[::-1]
            for j in range(hmin[0].size):
                hmin[i][j] = min(hl[j], hr[j])
                hmax[i][j] = max(hl[j], hr[j])
        for k in range(len(xs)):
            plt.vlines(nb[k], hmin[k], hmax[k], colors=cs[k], alpha=1, linewidth=2)
                
#        for i in range(len(xs)):
#            h = plt.hist(xs[i], bins=nb[i], label=labels[i], alpha=1,
#                         edgecolor=cs[i], linewidth=2, fill=False,
#                         cumulative=cumulative)
#            h_list_left = list(h[0])
#            h_list_left.append(0)
#            h_list_right = list(h[0][::-1])
#            h_list_right.append(0)
#            hl = np.array(h_list_left)
#            hr = np.array(h_list_right)[::-1]
#            hh = np.zeros(hl.size, dtype='float')
#            for j in range(hh.size):
#                hh[j] = min(hl[j], hr[j])
#            plt.vlines(nb[i], 0, hh, colors='white', alpha=1, linewidth=2)
    else:
        for i in range(len(xs)):
            h = plt.hist(xs[i], bins=nb[i], label=labels[i], alpha=a[i],
                         color=cs[i], fill=True, cumulative=cumulative)
                         
                         
#    for i in range(len(xs)):
#        if edge==True:
##            h = plt.hist(xs[i], bins=nb[i], label=labels[i], alpha=1,
##                         edgecolor=cs[i], linewidth=2, fill=False,
##                         cumulative=cumulative)
##            hlist = list(h[0])
##            hlist.append(0)
##            h = np.array(hlist)
##            print(h, h.size, type(h), type(h[0]))
##            print()
##            plt.vlines(nb[i], 0, h, colors='white', alpha=1, linewidth=2)
#
#            h = plt.hist(xs[i], bins=nb[i], label=labels[i], alpha=1,
#                         edgecolor=cs[i], linewidth=2, fill=False,
#                         cumulative=cumulative)
#            h_list_left = list(h[0])
#            h_list_left.append(0)
#            h_list_right = list(h[0][::-1])
#            h_list_right.append(0)
#            hl = np.array(h_list_left)
#            hr = np.array(h_list_right)[::-1]
#            hh = np.zeros(hl.size, dtype='float')
#            for j in range(hh.size):
#                hh[j] = min(hl[j], hr[j])
#            plt.vlines(nb[i], 0, hh, colors='white', alpha=1, linewidth=2)
#        else:
#            h = plt.hist(xs[i], bins=nb[i], label=labels[i], alpha=a[i],
#                         color=cs[i], fill=True, cumulative=cumulative)
    
    plt.title(title, size=24)
    plt.xlabel(xlabel, size=20)
    plt.ylabel(ylabel, size=20)
    plt.xlim(xlim)
    plt.yscale(yscale)
    plt.legend(loc=loc, fontsize=fs)
    plt.show()
    
    
    
def stats(x, label=None, misc=False, mins=False, maxs=False, alll=False,
                    minlim=None, maxlim=None, nextlowest=0, nexthighest=0):
    '''
    This function prints out important statistics that give a better sense of
    the distribution.
    
    Inputs:
        - x (1D array)     : distribution to analyze
    Kwargs:
        - label (str)      : labels the distribution (e.g. '2 GeV')
        - misc (bool)      : whether or not to print miscellaneous statistics
        - mins (bool)      : whether or not to print minimum statistics
        - maxs (bool)      : whether or not to print maximum statistics
        - alll (bool)      : whether or not to print all statistics; overrides
                             False assignments of variables above
        - minlim (float)   : elements below this number are shown and
                             investigated
        - maxlim (float)   : elements above this number are shown and
                             investigated
        - nextlowest (int) : the (n+1)-lowest value in the distribution will
                             be shown and enumerated
        - nexthighest (int): the (n+1)-highest value in the distribution will
                             be shown and enumerated
        
    Prints:
        - Many strings containing statements about the distribution
    '''
    if alll==True:
        misc = True
        mins = True
        maxs = True
    
    if label!=None:
        print(label)
        print()
        
    if misc==True:
        print('Length: {}'.format(x.size))
        print('Median: {}'.format(np.median(x)))
        print('Standard deviation: {}'.format(np.std(x)))
        print()
        print()
        
    if mins==True:
        print('Minimum of distribution: {}'.format(x.min()))
        print('Number of elements with this minimum: {}'.format(x[x==x.min()].size))
        print('10th percentile of distribution: {}'.format(np.quantile(x, 0.1)))
        print()
        if nextlowest!=0:
            xcp = copy.deepcopy(x)
            xcp = np.where(xcp==xcp.min(), xcp.max(), xcp)
            for i in range(nextlowest):
                print('Next lowest value is: {}'.format(xcp.min()))
                print('Number of elements with this value: {}'.format(xcp[xcp==xcp.min()].size))
                print()
                xcp = np.where(xcp==xcp.min(), xcp.max(), xcp)
        if minlim!=None:
            print('Number of elements below {}: {}'.format(minlim,
                                                            x[x<minlim].size))
            print('Corresponding percentile: {}'.format(x[x<minlim].size / x.size))
            print('Array of elements below {}: {}'.format(minlim,
                                                np.sort(x[x<minlim])[::-1]))
            print()
        print()
            
    if maxs==True:
        print('Maximum of distribution: {}'.format(x.max()))
        print('Number of elements with this maximum: {}'.format(x[x==x.max()].size))
        print('90th percentile of distribution: {}'.format(np.quantile(x, 0.9)))
        print()
        if nexthighest!=0:
            xcp = copy.deepcopy(x)
            xcp = np.where(xcp==xcp.max(), xcp.min(), xcp)
            for i in range(nexthighest):
                print('Next highest value is: {}'.format(xcp.max()))  #HERE
                print('Number of elements with this value: {}'.format(xcp[xcp==xcp.max()].size))
                print()
                xcp = np.where(xcp==xcp.max(), xcp.min(), xcp)
        if maxlim!=None:
            print('Number of elements above {}: {}'.format(maxlim,
                                                            x[x>maxlim].size))
            print('Corresponding percentile: {}'.format(x[x<maxlim].size / x.size))
            print('Array of elements above {}: {}'.format(maxlim,
                                                        np.sort(x[x>maxlim])))
            print()
        print()
    
    
    
def calc_bins_multi(alist, nbins=15):
    '''
    Returns the bins to be used in a histogram or profile plot
    
    Inputs:
        - alist (list) : 2D list where each element is a 1D distribution
    Kwargs:
        - nbins (int)  : starting number of desired bins
        
    Returns:
        - (numpy array): array of bins to be used in histogram or profile
                         plot
    '''
    bins = np.zeros((len(alist), nbins+1), dtype='float')
    
    for i in range(len(alist)):
        h, bins[i] = np.histogram(alist[i], bins=nbins)
        
#    for i in range(bins.shape[0]):
#        if bins[i,0]==bins[:,0].min() and bins[i,-1]==bins[:,-1].max():
#            return bins[i]
            
    bmin = np.where(bins==bins.min())
    bmax = np.where(bins==bins.max())
    imin = bmin[0][0]
    imax = bmax[0][0]
    
    if imin==imax:
        return bins[imin]
    
    rbins = bins[imin]
    rbins_list = list(rbins)
    
    while rbins_list[-1] < bins.max():
        diff = rbins_list[-1] + rbins[1]-rbins[0]
        rbins_list.append(diff)
        
    rbins_array = np.array(rbins_list)
    return rbins_array
    
    
    
    
'''
These are the functions needed to graph the 2D scatterplot of binned medians
for hit energies and hit positions (also in graph_functions.py).
'''

def calc_bins(a1, a2, nbins=15):
    '''
    Returns the bins to be used in the 2D scatterplot
    
    Inputs:
        - a1 (numpy array)    : coordinate array for first neutron energy
                                (e.g. 1 GeV)
        - a2 (numpy array)    : coordinate array for second neutron energy
                                (e.g. 2 GeV)
    Kwargs:
        - nbins (int)         : starting number of desired bins
        
    Outputs:
        - (bins) (numpy array): array of bins to be used in histogram
    '''
    
    h1, bins1 = np.histogram(a1, bins=nbins)
    h2, bins2 = np.histogram(a2, bins=nbins)
    
    
    if bins1[0]<=bins2[0] and bins1[-1]>=bins2[-1]:
        return bins1
    
    
    elif bins2[0]<=bins1[0] and bins2[-1]>=bins1[-1]:
        return bins2
    
    
    elif bins1[0]<=bins2[0]:
        bins1_list = list(bins1)
        
        while bins1_list[-1]<bins2[-1]:
            diff = bins1_list[-1] + (bins1[1]-bins1[0])
            bins1_list.append(diff)
            
        bins1_array = np.array(bins1_list)
        return bins1_array
    
    
    elif bins2[0]<=bins1[0]:
        bins2_list = list(bins2)
        
        while bins2_list[-1]<bins1[-1]:
            diff = bins2_list[-1] + (bins2[1]-bins2[0])
            bins2_list.append(diff)
            
        bins2_array = np.array(bins2_list)
        return bins2_array



def ms_and_qs(xy, bins, low=0.16, high=0.84):
    '''
    Calculates median and middle "68% quantile" for each bin of the 2D
    array containing (coordinates, hit energies)
    (Can also calculate other quantile ranges)
    
    Inputs:
        - xy (numpy array)     : aformentioned 2D array
        - bins (numpy array)   : output from calc_bins function, the binning
                                 to be used in the future histogram
    Kwargs:
        - low (float in [0,1]) : lower percentile for quantile range
        - high (float in [0,1]): higher percentile for quantile range
                              
    Outputs:
        - medians (numpy array): 2D array containing bin medians of
                                 coordinates and energies
        - q_low (numpy array)  : 2D array containing 16th quartile value of
                                 coordinates and energies
        - q_high (numpy array) : 2D array containing 84th quartile value of
                                 coordinates and energies
    '''
    
    nb = bins.size
    
    medians = np.zeros((nb-1, 2), dtype='float')
    q_low   = np.zeros((nb-1, 2), dtype='float')
    q_high  = np.zeros((nb-1, 2), dtype='float')
    
    
    for i in range(nb-1):
    
        xx = xy[:,0][xy[:,0]>=bins[i]]
        xxx = xx[xx<bins[i+1]]
    
        if xxx.size==0:
            medians[i,0] = bins[i]
            q_low[i,0] = 0
            q_high[i,0] = 0
        else:
            medians[i,0] = np.median(xxx)
            q_low[i,0] = medians[i,0] - np.quantile(xxx, low)
            q_high[i,0] = np.quantile(xxx, high) - medians[i,0]
    
            if i==(nb-1):
                xx = xy[:,0][xy[:,0]>=bins[i]]
                xxx = xx[xx<=bins[i+1]]
                medians[i,0] = np.median(xxx)
                q_low[i,0] = medians[i,0] - np.quantile(xxx, low)
                q_high[i,0] = np.quantile(xxx, high) - medians[i,0]
                
                
    for i in range(nb-1):
    
        z1 = xy[:,0][xy[:,0]>=bins[i]]
        ez1 = xy[:,1][xy[:,0]>=bins[i]]

        zez1 = np.zeros((ez1.size, 2), dtype='float')
        zez1[:,0] = z1
        zez1[:,1] = ez1

        zz1 = zez1[:,1][zez1[:,0]<bins[i+1]]

        if zz1.size==0:
            medians[i,1] = 0
            q_low[i,1] = 0
            q_high[i,1] = 0
        else:
            medians[i,1] = np.median(zz1)
            q_low[i,1] = medians[i,1] - np.quantile(zz1, low)
            q_high[i,1] = np.quantile(zz1, high) - medians[i,1]

            if i==(nb-1):
                z1 = xy[:,0][xy[:,0]>=bins[i]]
                ez1 = xy[:,1][xy[:,0]>=bins[i]]

                zez1 = np.zeros((ez1.size, 2), dtype='float')
                zez1[:,0] = z1
                zez1[:,1] = ez1

                zz1 = zez1[:,1][zez1[:,0]<=bins[i+1]]

                medians[i,1] = np.median(zz1)
                q_low[i,1] = medians[i,1] - np.quantile(zz1, low)
                q_high[i,1] = np.quantile(zz1, high) - medians[i,1]
                
                
    return medians, q_low, q_high



def ms_qs_graph(medians, q_lows, q_highs, coord, energies, fs=12, ms=8,
                                    loc='best', fonts=20, colors=None):
    '''
    Graphs the medians and quantiles as returned from the function
    ms_and_qs
    
    Inputs:
        - medians (numpy array): 3D array where each 2D slice
                                 is the first output from ms_and_qs
        - q_lows (numpy array) : 3D array where each 2D slice
                                 is the second output from ms_and_qs
        - q_highs (numpy array): 3D array where each 2D slice
                                 is the third output from ms_and_qs
        - coord (str)          : name of the coordinate under analysis;
                                 takes values of 'x', 'y', 'z'
        - energies (list)      : contains energies of the fired neutrons
                                 for each 2D array slice
    Kwargs:
        - fs (int)             : size of graph (shape will always be square)
        - ms (int)             : size of data points
        - loc (str/int)        : location of the graph legend; 1 is upper
                                 right, 5 is middle right, 9 is top middle
        - fonts (int)          : font size of legend
        - colors (list)        : color of points and error bars for a
                                 particular dataset
                       
    Outputs:
        - matplotlib scatterplot of median positions and energies,
          along with their respective quantiles, for each position
          bin
    '''
    
    if colors != None:
        cs = colors
    else:
        cs = plt.cm.Set1(np.linspace(0, 1, medians.shape[0]))
    
#    colors = plt.cm.Set1(np.linspace(0, 1, medians.shape[0]))  # change the
#                                                # color scheme here if you like
    
    plt.figure(figsize=(fs,fs))
    plt.title('Median Hit Energy vs Median {}-Coordinates'.format(coord.upper()), size=20)
    plt.xlabel('median {}-coordinates of hit (mm)'.format(coord), size=18)
    plt.ylabel('median energy of hit (MeV)', size=18)
    
    for i in range(medians.shape[0]):
        plt.errorbar(medians[i,:,0], medians[i,:,1], xerr=q_lows[i,:,0],
                     yerr=q_lows[i,:,1], uplims=True, xuplims=True,
                     marker='o', ms=ms, label='{} GeV'.format(energies[i]),
                     lw=0, elinewidth=2, color=cs[i])
        plt.errorbar(medians[i,:,0], medians[i,:,1], xerr=q_highs[i,:,0],
                     yerr=q_highs[i,:,1], lolims=True, xlolims=True,
                     marker='o', ms=ms, lw=0, elinewidth=2, color=cs[i])
        
    plt.legend(loc=loc, fontsize=fonts)
    plt.show()

    
