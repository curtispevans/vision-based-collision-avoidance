import numpy as np

from pathplannerutility import *
import math
from scipy.optimize import minimize, NonlinearConstraint, differential_evolution
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import BSpline
# get colormap
ncolors = 256
color_array = plt.get_cmap('jet')(range(ncolors))

# change alpha values
color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)

# register this new colormap with matplotlib
plt.colormaps.register(cmap=map_object)



data = np.load('visualAvoidance2D/data/gaussian_map3.npy')
ownship = np.load('visualAvoidance2D/data/ownship_positions2.npy')

print(pdf_map_constraint(np.array([1000, 500]), data))