import numpy as np
import matplotlib.pyplot as plt
from utilities import WedgeEstimator

real = np.load('visualAvoidance2D/data/xplane_data/0001/all_positions_in_path.npy')
intruders = []

for intruder in real[1:]:
    intruders.append(intruder)

intruder_wingspan = 30

ownship = real[0]
# ownship = ownship[0] * np.ones_like(ownship)

bearings = []
sizes = []
rhos = []
for intruder_pos, ownship_pos in zip(intruders[0], ownship):
    bearing = np.arctan2(intruder_pos[0] - ownship_pos[0], intruder_pos[1] - ownship_pos[1])
    rho = np.linalg.norm(intruder_pos - ownship_pos)
    size = intruder_wingspan / rho
    bearings.append(bearing)
    sizes.append(size)
    rhos.append(rho)


for i, intruder in enumerate(intruders):
    plt.plot(intruder[:,0], intruder[:,1], 'ro', label=f"Intruder {i+1}")
plt.plot(ownship[:,0], ownship[:,1], 'bo', label='Ownship')
plt.show()

plt.subplot(3,1,1)
plt.plot(bearings, 'o')
plt.title('Bearings')

plt.subplot(3,1,2)
plt.plot(sizes, 'o')
plt.title('Sizes')

plt.subplot(3,1,3)
plt.plot(rhos, 'o')
plt.title('Rhos')
plt.tight_layout()
plt.show()

wedge = WedgeEstimator()
num = 10
wedge.set_velocity_position(bearings[:num], sizes[:num], [0.0]*num, ownship[:num].reshape(num,2,1))
ts = 1/30
num += 1
t = 0

while num < len(real[0]):
    plt.clf()
    vertices = wedge.get_wedge_vertices(t)
    own_pos = wedge.init_own_pos
    own_vel = wedge.init_own_vel
    pose = own_pos + own_vel * t
    plt.plot(ownship[num,0], ownship[num,1], 'bo', linewidth=1)
    for i, intruder in enumerate(intruders):
        plt.plot(intruder[num,0], intruder[num,1], 'ro', linewidth=1)
        mid_top = (vertices[0] + vertices[3]) / 2
        mid_bottom = (vertices[1] + vertices[2]) / 2
        ownship_reverse = np.array([ownship[num][1], ownship[num][0]]).reshape(-1,1)
        # mid_top += ownship_reverse
        # mid_bottom += ownship_reverse
        plt.plot([mid_top[1], mid_bottom[1]], [mid_top[0], mid_bottom[0]], 'go', linewidth=1)
        plt.plot([pose[0,0], mid_top[1,0]], [pose[1,0], mid_top[0,0]], 'k', linewidth=1)
    t += ts
    plt.xlim(-4000, 1100)
    plt.ylim(-10, 5000)
    plt.waitforbuttonpress()
    num += 1

plt.show()


print('\nmid_bottom', mid_bottom.flatten())
print('true', intruders[0][-1])
print('ownship', ownship[-1])