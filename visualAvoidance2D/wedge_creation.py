import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import animation

import utilities as utils
from pathplannerutility import *

# set the simulation time
sim_time = 0
start = time.time()

# set up ownship
ownship0 = utils.MsgState(pos=np.array([[0.0], [.0]]), vel=30, theta=0)
ownship = utils.MavDynamics(ownship0)

# load the intruder data
intruder1 = np.load('visualAvoidance2D/data/intruder1.npy')
intruder2 = np.load('visualAvoidance2D/data/intruder2.npy')
intruder3 = np.load('visualAvoidance2D/data/intruder3.npy')

# cut = 20
cut = 20
intruder1 = intruder1[cut:]
intruder2 = intruder2[cut:]
intruder3 = intruder3[cut:]

num_measurements = 25
window_size = 9
# num_measurements = 10
# window_size = 5

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
num_intruders = 3
for i in range(num_intruders):
    wedge_estimator = utils.WedgeEstimator()
    wedge_estimator.set_velocity_position(bearings_list[i], sizes_list[i], ownship_thetas, ownship_positions, ownship.state)
    wedges.append(wedge_estimator)

print(f"Initialized the wedges after {round(time.time() - start,2)} seconds")
start = time.time()

####################################################################################################################################
# Now we have the wedges initialized. We can get the GMM at any time and x, y by calling wedge_estimator.get_wedge(t).pdf(x, y)


plot = True

# testing the function
if plot:
    fig, ax = plt.subplots()# set the simulation time

zoom = 25000
x, y = np.linspace(-5000, 5000, 25), np.linspace(-1000, 9000, 25)
X, Y = np.meshgrid(x, y)

list_of_vertices = []

for i in range(25, 650):
    sim_time += utils.ts_simulation

    # get the sum of the wedges
    if i % 5 == 0:
        vertices = []
        for wedge in wedges:
            vertices.append(wedge.get_wedge_vertices(sim_time))
        list_of_vertices.append(vertices)
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

np.save('visualAvoidance2D/data/vertices.npy', np.array(list_of_vertices))


# now animate the plot
fig, ax = plt.subplots(dpi=300)
ax.set_xlim([-5000, 5000])
ax.set_ylim([-1000, 9000])
ax.set_aspect('equal')
# ax.set_title(f"Time = {round(sim_time,2)}")
ax.plot(ownship.state.pos.item(1), ownship.state.pos.item(0), 'bo')

def animate(i):
    ax.cla()
    for vertice in list_of_vertices[i]:
        utils.plot_wedge(vertice, ax)
    ax.plot(ownship.state.pos.item(1), ownship.state.pos.item(0), 'bo')
    ax.set_xlim([-5000, 5000])
    ax.set_ylim([-1000, 9000])
    ax.set_title(f"3 Intruders")
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    return ax

ani = animation.FuncAnimation(fig, animate, frames=len(list_of_vertices), interval=100)
ani.save('visualAvoidance2D/data/wedges.mp4')
plt.show()