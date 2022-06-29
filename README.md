# ldmx_graphs

I made this repository so that others would not have to start from scratch when creating graphs that analyze simulation data produced by ldmx-sw. 

## Jupyter Notebook

This notebook (100k_notebook.ipynb) is still in a bit of a rough state, but this is the code (graphs and array manipulations) I am using to better understand the simulation data produced by ldmx-sw.

## Other code

### Median Hit Energies vs Median Coordinates

The most documented functions currently take the user through the process of analyzing hit energy versus position within the detector (simulation data in 100k_1GeV.pkl and 100k_2GeV.pkl, functions in graph_functions.py, example of use in graph_ex.py). 

The process is as follows:

    1) Use the position data to determine the best binning (size/number of bins) with which to concurrently display both data sets.
    
    2) Within each bin, find the median and desired quantiles of that bin's positions and corresponding energies.
    
    3) Graph these medians and quantiles from both of these data sets concurrently.
    
Analyzing the graph produced by this example shows that, as expected, the 2 GeV neutrons have overall higher hit energies than the 1 GeV neutrons. This trend is especially strong in the core of the distribution, where there are higher quantities of hits.

### analysis_funcs.py

This file contains all of my functions that I use for graphing and understanding the simulation data. Along with the functions in graph_functions.py, it includes histogramming and scatterplot functions as well as a function that prints out important statistics about a distribution.
