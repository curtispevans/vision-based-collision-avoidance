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

# set the simulation time
sim_time = 0
start = time.time()

# set up ownship
ownship0 = utils.MsgState(pos=np.array([[-1000.0], [.0]]), vel=15, theta=0)
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
num_intruders = 2
for i in range(num_intruders):
    wedge_estimator = utils.WedgeEstimator()
    wedge_estimator.set_velocity_position(bearings_list[i], sizes_list[i], ownship_thetas, ownship_positions, ownship.state)
    wedges.append(wedge_estimator)

print(f"Initialized the wedges after {round(time.time() - start,2)} seconds")
start = time.time()

plot = False

# testing the function
if plot:
    fig, ax = plt.subplots()# set the simulation time

zoom = 25000
x, y = np.linspace(-5000, 5000, 25), np.linspace(-1000, 9000, 25)
X, Y = np.meshgrid(x, y)

list_of_vertices = []
pdf_map_list = []

for i in range(25, 650):
    sim_time += utils.ts_simulation

    # get the sum of the wedges
    if i % 25 == 0:
        def pdf(xy,sim_time=sim_time):
            return sum([wedge.get_wedge_single_gaussian(sim_time).pdf(xy) for wedge in wedges])
        
        # Z = pdf(np.dstack((Y,X)))
        # pdf_map_list.append(Z)

        Z = np.zeros((25, 25))
        vertices = []
        points = np.vstack((Y.ravel(), X.ravel())).T
        for wedge in wedges:
            vertice = wedge.get_wedge_vertices(sim_time)
            vertices.append(vertice)
            Z += utils.are_inside_wedge(points, vertice).reshape(25, 25)
        list_of_vertices.append(vertices)

        pdf_map_list.append(Z)
        if plot:
            ax.cla()
            for vertice in vertices:
                utils.plot_wedge(vertice, ax)
            ax.plot(ownship.state.pos.item(1), ownship.state.pos.item(0), 'bo')
            ax.set_xlim([-5000, 5000])
            ax.set_ylim([-1000, 9000])
            ax.set_title(f"Time = {round(sim_time,2)}")
            plt.pause(0.01)

if plot:
    plt.show()




# get colormap
ncolors = 256
color_array = plt.get_cmap('jet')(range(ncolors))

# change alpha values
color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)

# register this new colormap with matplotlib
plt.colormaps.register(cmap=map_object)

data = np.array(pdf_map_list)

original_shape = data.shape
print("Original shape:", original_shape)

scale_resolution = 1
probability_threshold = num_intruders*.6e-8
new_shape = (25, 25, 25)


print("Downsampled shape:", data.shape)

start = (0, 0, 13)
goal = (new_shape[0]-1, new_shape[1]-1, 13)
print("start point:",start, "goal point:", goal)

binary_matrix = binarize_matrix(data, 5e-8)
print(binary_matrix.shape)

# path = [(i, i, 15) for i in range(25)]
# print(data)


path = bidirectional_a_star(data, start, goal)

print("path:", path)
print("path length:", len(path))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

voxels_transposed = np.transpose(data, (2, 1, 0))
ax.voxels(voxels_transposed, edgecolor='none', alpha=0.5)

for i in range(0, len(path)):
    ax.plot(path[i][2], path[i][1], path[i][0], 'o', color='blue', markersize=4)

plt.show()

int_X0 = []
past = -1
scaler_shift = 10000/new_shape[1]
for i in range(0, len(path)):
    if path[i][0] != past:
        int_X0.append(scaler_shift*path[i][1]-1000)
        int_X0.append(scaler_shift*path[i][2]-5000)
    past = path[i][0]

array_X0 = np.array(int_X0).reshape(len(int_X0)//2, 2).T
new_X0 = []

for i in range(0, len(int_X0)-2, 2):
    new_X0.append(int_X0[i])
    new_X0.append(int_X0[i+1])
    new_X0.append((int_X0[i] + int_X0[i+2])/2)
    new_X0.append((int_X0[i+1] + int_X0[i+3])/2)

new_X0.append(int_X0[-2])
new_X0.append(int_X0[-1])

print("int_X0:", int_X0)
print("len(int_X0):", len(int_X0))
start_point = np.array([scaler_shift*start[1]-1000, scaler_shift*start[2]-5000])
print("start_point:", start_point)
goal_point = np.array([scaler_shift*goal[1]-1000, scaler_shift*goal[2]-5000])
print("goal_point:", goal_point)

int_X0 = np.array(int_X0)
print('Starting optimization...')
start = time.time()
nlc = NonlinearConstraint(lambda x: con_cltr_pnt(x, start_point), 0.0, 400.)

array_of_vertices = np.array(list_of_vertices).reshape(25,2,4,2)
bounds_for_optimization = None
res = minimize(objective_function_with_constraints, int_X0, args=(np.array([goal_point[0], goal_point[1]]),array_of_vertices,), method='SLSQP', jac='2-point', bounds=bounds_for_optimization, options={'maxiter':500, 'disp':True}, constraints=[nlc,], )

print(f'Optimization done in {round(time.time() - start,2)} seconds')
print(res.success)
print(res.message)
print(len(res.x))


# Show the plot
z = np.arange(data.shape[0])
x, y = np.linspace(-5000, 5000, 25), np.linspace(-1000, 9000, 25)

X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, vertices in enumerate(list_of_vertices):
    for vertice in vertices:
        ax.plot([vertice[0][1], vertice[1][1]], [vertice[0][0], vertice[1][0]], i, color='red')
        ax.plot([vertice[1][1], vertice[2][1]], [vertice[1][0], vertice[2][0]], i, color='red')
        ax.plot([vertice[2][1], vertice[3][1]], [vertice[2][0], vertice[3][0]], i, color='red')
        ax.plot([vertice[3][1], vertice[0][1]], [vertice[3][0], vertice[0][0]], i, color='red')


x, y = np.linspace(-5000, 5000, 25), np.linspace(-1000, 9000, 25)

X, Y = np.meshgrid(x, y)



# for i in range(0, data.shape[0]):
   
#    sc =  ax.contourf(X, Y, data[i, :, :], 100, zdir='z', offset=i, cmap='binary')
# cbar = plt.colorbar(sc, ax=ax, pad=0.1)
# cbar.set_label('Color Scale')

for i in range(0, len(int_X0)-2, 2):
    ax.plot(int_X0[i+1], int_X0[i], int(i/2), 'o', color='blue', markersize=4, alpha=0.5)
    ax.plot(res.x[i+1], res.x[i], int(i/2), 'o', color='green', markersize=4)
    
ax.plot(int_X0[-1], int_X0[-2], 24, 'o', color='blue', markersize=4, label='Initial Path', alpha=0.5)
ax.plot(res.x[-1], res.x[-2], 24, 'o', color='green', markersize=4, label='Optimal Path')

    
# voxels_transposed = np.transpose(data, (2, 1, 0))
# ax.voxels(voxels_transposed, edgecolor='none', alpha=0.5)

ax.set_zlim(0, new_shape[0])
ax.set_xlim([-5000, 5000])
ax.set_ylim([-1000, 9000])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Time axis')
ax.legend()

title = "visualAvoidance2D/figures/success_two_across"
# plt.savefig(title+'.png')

plt.show()


fig, ax = plt.subplots()

x, y = np.linspace(-5000, 5000, 200), np.linspace(-1000, 9000, 200)

X, Y = np.meshgrid(x, y)

print("animating optimal path")
def update(frame):
    ax.cla()
    ax.scatter(res.x[2*frame+1], res.x[2*frame])
    for vertice in list_of_vertices[frame]:
        utils.plot_wedge(vertice, ax)
    ax.set_xlim([-5000, 5000])
    ax.set_ylim([-1000, 9000])

    
    return ax

ani = animation.FuncAnimation(fig, update, frames=range(len(res.x)//2), repeat=False)

# Save the animation as an MP4 file
# ani.save(title+'.mp4', writer='ffmpeg', fps=10)
plt.show()



