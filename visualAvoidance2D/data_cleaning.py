import numpy as np
import matplotlib.pyplot as plt

# data = np.load('visualAvoidance2D/data/xplane_data/0003/20241205_152441_bearing_info.npy')
data = np.load('visualAvoidance2D/data/xplane_data/0001/bearing_info.npy')
# all_positions = np.load('visualAvoidance2D/data/xplane_data/0003/20241205_152441_all_positions_in_path.npy')
all_positions = np.load('visualAvoidance2D/data/xplane_data/0001/all_positions_in_path.npy')

print(all_positions[1, :30, :])
print(np.diff(all_positions[1, :30, :], axis=0) / (1/30))
colors = ['ro', 'go', 'yo']
for i, position in enumerate(all_positions[1:]):
    print(i)
    plt.plot(position[:,0], position[:,1], colors[i], label=f"Intruder {i+1}")

plt.plot(all_positions[0][:,0], all_positions[0][:,1], 'bo', label='Ownship')
plt.legend()

plt.show()

bearings = [[] for _ in range(data.shape[1])]
sizes = [[] for _ in range(data.shape[1])]
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        bearings[j].append(data[i][j,0])
        sizes[j].append(data[i][j,1])

plt.plot(sizes[0][:30], 'o')
plt.show()

plt.plot(bearings[0][:30], 'o')
plt.show()

plt.plot(sizes[1][:30], 'o')
plt.show()

plt.plot(sizes[2][:30], 'o')
plt.show()

