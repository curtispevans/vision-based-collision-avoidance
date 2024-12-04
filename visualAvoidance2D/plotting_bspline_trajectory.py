import matplotlib.pyplot as plt
import numpy as np
import math
import time
from scipy.optimize import minimize, NonlinearConstraint, differential_evolution
from matplotlib import cm
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import BSpline
from scipy import ndimage

import utilities as utils
from pathplannerutility import *

# control_points = np.load('visualAvoidance2D/data/control_points.npy')

def get_knot_points(num_control_points, degree):
    num_knots = num_control_points + degree + 1
    knots = np.concatenate((np.zeros(degree), np.linspace(0,1,num_knots-2*degree), np.ones(degree)))
    return knots

k = 3
c = np.array([[-1,-1],[0,0], [3,3], [0,5], [-2,4], [-3,2], [-1,0]])
knots = get_knot_points(len(c), k)
spl = BSpline(knots, c, k)
t = np.linspace(0, 1, 100)
curve = spl(t)
plt.plot(c[:,0], c[:,1], 'bo', lw=4, alpha=0.7, label='BSpline')
plt.plot(curve[:,0], curve[:,1], 'r-', lw=1, alpha=0.7, label='BSpline')
plt.show()
