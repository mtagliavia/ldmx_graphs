'''
These are the functions needed to graph the 2D scatterplot of binned medians for hit energies and hit positions
'''


def calc_bins(a1, a2, nbins=15):
    '''
    Returns the bins to be used in the 2D scatterplot
    
    Inputs:
        - a1 (numpy array): coordinate array for first neutron energy (e.g. 1 GeV)
        - a2 (numpy array): coordinate array for second neutron energy (e.g. 2 GeV)
        - nbins (int): starting number of desired bins
        
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
            diff = bins2_list[-1] + (bins2[1]-bins1[0])
            bins2_list.append(diff)
            
        bins2_array = np.array(bins2_list)
        return bins2_array



def ms_and_qs(xy, bins, low=0.16, high=0.84):
    '''
    Calculates median and middle "68% quantile" for each bin of the 2D
    array containing (coordinates, hit energies)
    (Can also calculate other quantile ranges)
    
    Inputs:
        - xy (numpy array): aformentioned 2D array
        - bins (numpy array): output from calc_bins function, the binning
                              to be used in the future histogram
        - low (float in [0,1]): lower percentile for quantile range
        - high (float in [0,1]): higher percentile for quantile range
                              
    Outputs:
        - medians (numpy array): 2D array containing bin medians of
                                 coordinates and energies
        - q_low (numpy array): 2D array containing 16th quartile value of
                               coordinates and energies
        - q_high (numpy array): 2D array containing 84th quartile value of
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

            if i==(nbins-1):
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



def ms_qs_graph(medians, q_lows, q_highs, coord, energies, fs=12, ms=8, loc='best'):
    '''
    Graphs the medians and quantiles as returned from the function
    ms_and_qs
    
    Inputs:
        - medians (numpy array): 3D array where each 2D slice
                                 is the first output from ms_and_qs
        - q_lows (numpy array): 3D array where each 2D slice
                                is the second output from ms_and_qs
        - q_highs (numpy array): 3D array where each 2D slice
                                 is the third output from ms_and_qs
        - coord (str): name of the coordinate under analysis; takes
                       values of 'x', 'y', 'z'
        - energies (list): contains energies of the fired neutrons
                           for each 2D array slice
                       
    Outputs:
        - matplotlib scatterplot of median positions and energies,
          along with their respective quantiles, for each position
          bin
    '''
    
    colors = plt.cm.Set1(np.linspace(0, 1, medians.shape[0]))  # change the color scheme here if you like
    
    plt.figure(figsize=(fs,fs))
    plt.title('Median Hit Energy vs Median {}-Coordinates'.format(coord.upper()))
    plt.xlabel('median {}-coordinates of hit (mm)'.format(coord))
    plt.ylabel('median energy of hit (MeV)')
    
    for i in range(medians.shape[0]):
        plt.errorbar(medians[i,:,0], medians[i,:,1], xerr=q_lows[i,:,0], yerr=q_lows[i,:,1], uplims=True, xuplims=True,
                     marker='o', ms=ms, label='{} GeV'.format(energies[i]), lw=0, elinewidth=2, color=colors[i])
        plt.errorbar(medians[i,:,0], medians[i,:,1], xerr=q_highs[i,:,0], yerr=q_highs[i,:,1], lolims=True, xlolims=True,
                     marker='o', ms=ms, lw=0, elinewidth=2, color=colors[i])
        
    plt.legend(loc=loc)
