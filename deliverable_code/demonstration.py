from path_planner import plan_path
import numpy as np
from matplotlib import pyplot as plt
import time
import params

def get_bearing_size_measurements(filepath):
    data = np.load(filepath)
    bearings = [[] for _ in range(data.shape[1])]
    sizes = [[] for _ in range(data.shape[1])]
    for i in range(params.measured_window):
        for j in range(data.shape[1]):
            bearings[j].append(data[i][j,0])
            sizes[j].append(data[i][j,1])
    return bearings, sizes

def get_ownship_intruder_positions(filepath):
    real = np.load(filepath)
    ownship = real[0]
    intruders = []
    for intruder in real[1:]:
        intruders.append(intruder)
    return ownship, intruders

def plot_control_points(cntrl_pts):
    for i in range(0, len(cntrl_pts), 2):
        plt.plot(cntrl_pts[i+1], cntrl_pts[i], 'ko')
    
    plt.axis('equal')
    plt.show()


# Example usage of the path planner
bearings, sizes = get_bearing_size_measurements("deliverable_code/xplane_data/0002/20241205_151830_bearing_info.npy")
ownship, intruders = get_ownship_intruder_positions("deliverable_code/xplane_data/0002/20241205_151830_all_positions_in_path.npy")

start = time.time()
cntrl_pts = plan_path(bearings, sizes, ownship, len(intruders), -150)
end = time.time()
print("\nTime taken to plan path: ", end - start, " seconds\n")
plot_control_points(cntrl_pts)



