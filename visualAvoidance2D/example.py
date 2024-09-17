import matplotlib.pyplot as plt
import numpy as np
import math
import time
from scipy.optimize import minimize, NonlinearConstraint, differential_evolution
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import BSpline

import utilities as utils
from pathplannerutility import *

# set the simulation time
sim_time = 0
start = time.time()

# set up ownship
ownship0 = utils.MsgState(pos=np.array([[1306.127], [529.102]]), vel=15, theta=0)
ownship = utils.MavDynamics(ownship0)

# load the intruder data
intruder1 = np.load('visualAvoidance2D/data/intruder1.npy')
intruder2 = np.load('visualAvoidance2D/data/intruder2.npy')
intruder3 = np.load('visualAvoidance2D/data/intruder3.npy')

cut = 20
intruder1 = intruder1[cut:]
intruder2 = intruder2[cut:]
intruder3 = intruder3[cut:]

num_measurements = 25
window_size = 9

mf1 = utils.MedianFilter(window_size)
mf2 = utils.MedianFilter(window_size)
mf3 = utils.MedianFilter(window_size)

# filter the size data
mf1.fit(list(intruder1[:window_size-1,1]))
mf2.fit(list(intruder2[:window_size-1,1]))
mf3.fit(list(intruder3[:window_size-1,1]))

shift_index = int(np.median(np.arange(window_size)))

# get bearing lists
bearings_list = [intruder1[shift_index:num_measurements,0], intruder2[shift_index:num_measurements,0], intruder3[shift_index:num_measurements,0]]

intruder1_size = []
intruder2_size = []
intruder3_size = []

for i in range(window_size-1, num_measurements+shift_index):
    intruder1_size.append(mf1.predict(intruder1[i,1]))
    intruder2_size.append(mf2.predict(intruder2[i,1]))
    intruder3_size.append(mf3.predict(intruder3[i,1]))

# get the size lists
sizes_list = [intruder1_size, intruder2_size, intruder3_size]

# save the ownship positions and headings
ownship_positions = []
ownship_thetas = []

for i in range(shift_index):
    ownship.update(0)
    sim_time += utils.ts_simulation

for i in range(shift_index, num_measurements):
    ownship_positions.append(ownship.state.pos)
    ownship_thetas.append(ownship.state.theta)
    ownship.update(0)
    sim_time += utils.ts_simulation

# now initialize the wedges
wedges = []
for i in range(3):
    wedge_estimator = utils.WedgeEstimator()
    wedge_estimator.set_velocity_position(bearings_list[i], sizes_list[i], ownship_thetas, ownship_positions, ownship.state)
    wedges.append(wedge_estimator)

####################################################################################################################################
# Now we have the wedges initialized. We can get the GMM at any time and x, y by calling wedge_estimator.get_wedge(t).pdf(x, y)

# to get the sum we need to so something like this
def get_wedge_sum(t, xy, wedges=wedges):
    return sum([wedge.get_wedge(t).pdf(xy) for wedge in wedges])

plot = False

# testing the function
if plot:
    fig, ax = plt.subplots()

zoom = 25000
x, y = np.linspace(-12500, 12500, 100), np.linspace(0, zoom, 100)
X, Y = np.meshgrid(x, y)

pdf_funcs = []
pdf_map = []
for i in range(25, 525):
    sim_time += utils.ts_simulation

    # get the sum of the wedges
    if i % 5 == 0:
        def pdf(xy,sim_time=sim_time):
            return sum([wedge.get_wedge(sim_time).pdf(xy) for wedge in wedges])

        pdf_funcs.append(pdf)

        Z = get_wedge_sum(sim_time, np.dstack((Y, X)))
        pdf_map.append(Z)
        
        if plot:
            ax.cla()
            ax.contour(X, Y, Z, levels=10)
            ax.set_xlim([-12500, 12500])
            ax.set_ylim([0, zoom])
            plt.pause(0.01)

if plot:
    plt.show()

print(f'Saved the list of functions and 3D map after {round(time.time() - start,2)} seconds')
# print(len(pdf_funcs[::4]))

############################################################################################
# running path planner 
# Code written by JJ
############################################################################################

# get colormap
ncolors = 256
color_array = plt.get_cmap('jet')(range(ncolors))

# change alpha values
color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)

# register this new colormap with matplotlib
plt.colormaps.register(cmap=map_object)

data = np.array(pdf_map)

original_shape = data.shape
print("Original shape:", original_shape)

scale_resolution = 1
probability_threshold = 1e-10
new_shape = (25, 25, 25)
reshaped_data = data.reshape(new_shape[0], original_shape[0]//new_shape[0],
                             new_shape[1], original_shape[1]//new_shape[1],
                             new_shape[2], original_shape[2]//new_shape[2])

downsampled_data = reshaped_data.mean(axis=(1, 3, 5))
data = downsampled_data

print("Downsampled shape:", downsampled_data.shape)

start = (0, 0, 14)
goal = (24, 24, 15)
print("start point:",start, "goal point:", goal)

binary_matrix = binarize_matrix(data, 0.00000001)
path = bidirectional_a_star(binary_matrix, start, goal)

print("path:", path)
print("path length:", len(path))

int_X0 = []
past = -1
for i in range(0, len(path)):
    # print(i,path[i])
    if path[i][0] != past:
        int_X0.append(path[i][1]*scale_resolution)
        int_X0.append(path[i][2]*scale_resolution)
    past = path[i][0]

print("int_X0:", len(int_X0))
start_point = [start[1], start[2]]
print("start_point:", start_point)
goal_point = [goal[1], goal[2]]
print("goal_point:", goal_point)

print('Starting optimization...')
nlc = NonlinearConstraint(lambda x: con_cltr_pnt(x, start_point), 0.0, 1.0)
P_nlc = NonlinearConstraint(lambda x: pdf_map_constraint_functionized(x, functions=pdf_funcs[::4]), 0.0, probability_threshold)
res = minimize(object_function, int_X0, args=((goal_point[0], goal_point[1]),), method='SLSQP', bounds=[(-1, 26) for i in range(len(int_X0))], constraints=[nlc, P_nlc])

print('Optimization done!')
print(res.success)
print(res.message)
print(len(res.x))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Show the plot
z = np.arange(data.shape[0])
y = np.arange(data.shape[1])
x = np.arange(data.shape[2])

x, y = np.meshgrid(x, y)
voxels_transposed = np.transpose(binary_matrix, (2, 1, 0))
ax.voxels(voxels_transposed, edgecolor='none', alpha=0.1)

for i in range(0, data.shape[0]):
   sc =  ax.contourf(x, y, data[i, :, :], 100, zdir='z', offset=i, cmap='rainbow_alpha')
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Color Scale')


# Plot the path
z_coords = [point[0] for point in path]
x_coords = [point[2] for point in path]
y_coords = [point[1] for point in path]
ax.plot(x_coords, y_coords, z_coords, label='Path', color='green', linewidth=3, zorder=1)

for i in range(0, len(int_X0), 2):
    ax.scatter3D(int_X0[i+1], int_X0[i], int(i/2))


ax.set_zlim(0, 25)
ax.set_ylim(0, 25)
ax.set_xlim(0, 25)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Time axis')

plt.savefig('path_with_func_constraints_1e_neg10.png')
plt.show()