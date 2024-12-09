import numpy as np
from utilities import MedianFilter, WedgeEstimator

def create_wedges(filepath_bearing, filepath_true, num_measurements, window_size):
    '''Given a file path, this function will return a list of wedge estimators.'''
    data = np.load(filepath_bearing)
    ownship_real = np.load(filepath_true)[0]
    num_intruders = data.shape[1]
    wedges = [WedgeEstimator() for _ in range(num_intruders)]
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

    for i in range(shift_index, num_measurements):
        for j in range(num_intruders):
            bearings_list[j].append(data[i][j,0])
            sizes_list[j].append(median_filters[j].predict(data[i+shift_index-1][j,1]))
    
    ownship_positions = []
    # ownship_thetas = []
    for i in range(shift_index, num_measurements):
        ownship_positions.append(ownship_real[i])
        # ownship_thetas.append(ownship_real[i,2])
    

def test():
    filepath_bearing = 'visualAvoidance2D/data/xplane_data/0003/20241205_152441_bearing_info.npy'
    filepath_true = 'visualAvoidance2D/data/xplane_data/0003/20241205_152441_all_positions_in_path.npy'
    num_measurements = 25
    window_size = 9
    create_wedges(filepath_bearing, filepath_true, num_measurements, window_size)

if __name__ == '__main__':
    test()


