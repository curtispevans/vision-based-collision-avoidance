# Outline of Path Planner Code
## params.py
- `ts_simulation` is the frame rate it is set to $\frac{1}{30}$
- `min_area` is the minimum wingspan of the aircraft in the same airspace (currently set at 10 meaning the smallest aircraft we will see is a cessna)
- `max_ara` is the maximum winspan of the aircraft in the same airspace (currently set at 25 meaning the biggest aircraft we will see and try to avoid has a 25 meter wingspan)
- `bearing_uncertainty` is the bearing uncertainty when making the wedges it is in radians and should be kept relatively small.
- `measured_window` this is the number of frames used to calculate the wedges and ownship velocity. 
- `dim_astar` this is the dimension that the bidirectional A* is doing. It is set to 25 so there is a 25 x 25 x 25 voxel map. This is best for speed.
- `solution_span` this is the solution span distance so the path planner will solve for an area of `2*solution_span` x `2*solution_span`.

## wedge_utilities.py
This file contains the `WedgeEstimator` class and one helper function for the class. The initializer for the class saves the global variables.
- `WedgeEstimator.set_velocity_position(self, bearing_angles, sizes, ownship_positions)` takes in a list of bearing angles, list pixel sizes, and a list of the ownships positions being (east, north). This function then estimates the velocities needed to create each wedge and saves those velocities as member variables.
- `WedgeEstimator.get_wedge_vertices(self, t)` returns the 4 vertices of the wedge at a given time $t$. The vertices are returned in a (4, 2, 1) numpy array. With it being top right, bottom right, bottom left, and top left. All of these coordinates are in (north, east) format. 
- `get_wedge_vertices(t, close_pos, close_vel, far_pos, far_vel, bearing_uncertainty)` This computes the vertices for a given time $t$ and all the positions and velocities. This function is heavily dependent on the assumption that the intruders are traveling at a costant velocity. It returns a (4, 2, 1) numpy array with the order being top right, bottom right, bottom left, and top left with all the coordiantes being (north, east) format. 

## path_planner_utilities.py
- `are_inside_wedge(points, vertices)` this function checks if the points are inside the wedge formed by the vertices passes in. 
- `create_wedge(bearing_angles, sizes, ownship_positions)` this function creates a wedge estimator and sets the velocity and positions then returns the created wedge.
- `binarize_matrix(matrix, threshold)` this function creates a binary matrix with a given threshold. This is used for the A* algorithm.
- `heuristic(a,b)` returns the 2 norm of a and b. This is used for A*
- `get_neighbors(pos)` this function is used for A* 
- `bidirectional_a_star(grid, start, goal)` this function needs a (25,25,25) numpy array as the grid and then two tuples start and end of the form (time, North, East). These are all in the coordiantes of [0,25].
-`get_in_out_wedges_and_vertices(wedges)` this function takes in a list of wedges and will output the grid used for `bidirectional_a_star` and the list of vertices used for the optimization. 
- `con_cltr_pnt(x)` this function calculates the distance between each control point. It is used as one of the constraints in optimization. 
- `compute_cross_product2D(a, b)` this computes the 2D cross product of a and b.
- `compute_relavant_cross_products(p, vertices)` this computes the relevant cross products for the distance constraint function. It will return a value for right, bottom, left, top.
- `project_vector_onto_vector(a, b)` this projects the vector a onto the vector b.
- `compute_distance(a, b)` this computes the euclidean distance from vector a to b.
- `compute_distance_to_edge(point, v1, v2)` this computes the distance from a point to the edge of the wedge.
- `get_distance_from_edge_with_point_inside_wedge(p, vertices)` this computes the shortest distance from an edge for a point inside the wedge. 
- `distance_function(p, vertices)` this returns a distance value for a point inside or outside the wedge. It depends on where that point is oriented for which point/edge is used to compute the distance. 
- `distance_constraint_vectorized(x, vertice_list)` this is the function used in one of the constraints to ensure the control point is outside the wedge.
- `object_function_new(x, goalPosition)` this is the objective function for the optimization. It is minimizing the distance from the last control point to the goal point.
- `initialize_x0(path, start, end, dim)` this shifts all of the points in path returned from A* to be in the coordinate frame of ownship and not [0,25]. It will also remove any duplicated points in path. 
- `get_optimal_control_points(x0, start_point, end_point, array_of_vertices, safety_threshold)` this will return the optimal control points for the ownship. The safety threshold will push the control points farther away from the edge of the wedge. If the safety threshold is 0 that means it is on the edge of the wedge but if it is -100 is has to be -100 distance away from the wedge. 
-`get_knot_points(num_control_points, degree)` this will return the knot points for the b-spline.
- `get_bspline_path(control_points, degree)` this will return the curve or trajectory for ownship given the control points and a degree (which is usually 3).

## path_planner.py
- `plan_path(bearing_angles, sizes, ownship_poses, num_intruders, safety_threshold=-50)` this is the path planner function. It takes a list of bearings and sizes with the list format being [[intruder1_bearing1, intruder2_bearing2,...],[intruder2_bearing1, intruder2_bearing2,...],...]. It needs a list of ownship positions that match the dimension of a single intruders bearing or size list. The cooridantes are coded for (east, north) positions. It then needs the number of intruders at that time and a safety threshold. The safety threshold need to be negative. The smaller the threshold the safer the path planner will be. This function essentially calls all the other functions from utility files to create one nice function.  

## demonstration.py
This file contains a demo on how to use the path planner with the given xplane data. 