import numpy as np
from utilities import MedianFilter, WedgeEstimator
import matplotlib.pyplot as plt

def create_wedges(filepath_bearing, filepath_true, num_measurements, window_size):
    '''Given a file path, this function will return a list of wedge estimators.'''
    data = np.load(filepath_bearing)[:]
    ownship_real = np.load(filepath_true)[0]
    num_intruders = data.shape[1]
    median_filters = [MedianFilter(window_size) for _ in range(num_intruders)]
    median_filters_fit = [[] for _ in range(num_intruders)]

    shift_index = int(np.median(np.arange(window_size)))

    bearings_list = [[] for _ in range(num_intruders)]
    sizes_list = [[] for _ in range(num_intruders)]

    for time in data[:window_size-1]:
        for i in range(num_intruders):
            median_filters_fit[i].append(time[i,1])

    for i, median_filter in enumerate(median_filters):
        median_filter.fit(median_filters_fit[i]) 

    for i in range(0, num_measurements):
        for j in range(num_intruders):
            bearings_list[j].append(data[i][j,0])
            sizes_list[j].append(median_filters[j].predict(data[i+shift_index][j,1]))
            # sizes_list[j].append(data[i][j,1])
    
    ownship_positions = []
    ownship_thetas = []
    for i in range(1, num_measurements+1):
        pos = np.array([ownship_real[i-1][1], ownship_real[i-1][0]]).reshape(-1,1)
        ownship_positions.append(pos)
        theta = np.arctan2(ownship_real[i][0] - ownship_real[i-1][0], ownship_real[i][1] - ownship_real[i-1][1])
        ownship_thetas.append(0.0)


    wedges = []
    for i in range(num_intruders):
        print(i)
        wedge_estimator = WedgeEstimator()
        wedge_estimator.set_velocity_position(bearings_list[i], sizes_list[i], ownship_thetas, ownship_positions)
        wedges.append(wedge_estimator)

    return wedges

    
    

def test():
    # filepath_bearing = 'visualAvoidance2D/data/xplane_data/0003/20241205_152441_bearing_info.npy'
    filepath_bearing = 'visualAvoidance2D/data/xplane_data/0001/bearing_info.npy'
    # filepath_true = 'visualAvoidance2D/data/xplane_data/0003/20241205_152441_all_positions_in_path.npy'
    filepath_true = 'visualAvoidance2D/data/xplane_data/0001/all_positions_in_path.npy'
    real = np.load(filepath_true)[:,:,:]
    measurements = np.load(filepath_bearing)
    
    # print(ownship_real)
    num_measurements = 30
    window_size = 15
    wedges = create_wedges(filepath_bearing, filepath_true, num_measurements, window_size)
    ts = 1/30
    t = 0
    num = num_measurements+1
    while num < len(real[0]):
        # plt.clf()
        for i, wedge in enumerate(wedges):
            bearing = measurements[num, i, 0]
            size = measurements[num, i, 1]
            ownship_pos = real[0,num,:].reshape(-1,1)
            # vertices = wedge.get_wedge_and_update(t, bearing, size, 0.0, ownship_pos)
            vertices = wedge.get_wedge_vertices(t)
            plt.plot(real[i+1][num,0], real[i+1][num,1], 'ro')
            mid_top = (vertices[0] + vertices[3]) / 2
            mid_bottom = (vertices[1] + vertices[2]) / 2
            plt.plot([mid_top[1], mid_bottom[1]], [mid_top[0], mid_bottom[0]], 'go')
            # plt.waitforbuttonpress()
        t += ts
        plt.plot(real[0][num,0], real[0][num,1], 'bo')
        plt.xlim(-4000, 1000)
        plt.ylim(-10, 5000)
        # plt.pause(0.01)
        plt.waitforbuttonpress()
        num += 1

    plt.show()

def plot_wedge(vertices):
    plt.plot([vertices[0,1], vertices[1,1], vertices[2,1], vertices[3,1], vertices[0,1]],
             [vertices[0,0], vertices[1,0], vertices[2,0], vertices[3,0], vertices[0,0]], 'r', linewidth=2)
    
        

if __name__ == '__main__':
    test()


