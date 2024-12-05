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
c = np.load('visualAvoidance2D/data/optimal_path.npy').reshape(-1,2)

def get_bspline_path(control_points, degree):
    knots = get_knot_points(len(control_points), degree)
    spl = BSpline(knots, control_points, degree)
    t = np.linspace(0, 1, 100)
    curve = spl(t)
    return curve

# curve = get_bspline_path(c, k)
# # plotted as North East coordinates
# plt.plot(c[:,1], c[:,0], 'bo', lw=4, alpha=0.7, label='BSpline')
# plt.plot(curve[:,1], curve[:,0], 'r-', lw=1, alpha=0.7, label='BSpline')
# plt.xlim([-5000, 5000])
# plt.ylim([-1000, 9000])
# plt.show()
