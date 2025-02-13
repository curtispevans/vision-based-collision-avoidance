import numpy as np
from numpy.typing import NDArray
import params as params
import path_planner_utilities as ppu
from typing import List, Tuple

def plan_path(bearing_angles, sizes, ownship_poses, num_intruders, safety_threshold=-50) -> NDArray:
    wedges = []
    for i in range(num_intruders):
        wedge = ppu.create_wedge(bearing_angles[i], sizes[i], ownship_poses)
        wedges.append(wedge)

    in_out_wedge_list, list_of_vertices = ppu.get_in_out_wedges_and_vertices(wedges, plotting=False)

    data = np.array(in_out_wedge_list)
    binary_matrix = ppu.binarize_matrix(data, 0)
    start = (0, 0, 13) # (time, N, E)
    end = (24, 24, 13) # (time, N, E)
    path = ppu.bidirectional_a_star(binary_matrix, start, end)

    x0, start_point, end_point = ppu.initialize_x0(path, start, end, params.dim_astar)
    array_of_vertices = np.array(list_of_vertices).reshape(params.dim_astar, num_intruders,4,2)

    res = ppu.get_optimal_control_points(x0, start_point, end_point, array_of_vertices, safety_threshold)

    return res.x



    