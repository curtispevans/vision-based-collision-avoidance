import numpy as np
import matplotlib.pyplot as plt

data = np.load('visualAvoidance2D/data/xplane_data/0001/bearing_info.npy')
all_positions = np.load('visualAvoidance2D/data/xplane_data/0003/20241205_152441_all_positions_in_path.npy')

for position in all_positions:
    plt.plot(position[:,0], position[:,1], 'ro')

plt.show()