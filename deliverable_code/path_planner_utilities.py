import numpy as np
from numpy.typing import NDArray
import heapq
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.interpolate import BSpline
from wedge_utilities import WedgeEstimator
import params
from typing import List, Tuple



def are_inside_wedge(points : NDArray, vertices : NDArray) -> NDArray:
    # for this function to work a and b must be (2,) and o must be (n,2)
    v1, v2, v3, v4 = vertices
    def sign(a, b, o):
        # get the determinate of [a-o, b-o]
        return (a[0] - o[:,0]) * (b[1] - o[:,1]) - (b[0] - o[:,0]) * (a[1] - o[:,1])
    
    d1 = sign(v1, v2, points)
    d2 = sign(v2, v3, points)
    d3 = sign(v3, v4, points)
    d4 = sign(v4, v1, points)
    
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0) | (d4 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0) | (d4 > 0)
    
    return ~(has_neg & has_pos)

def create_wedge(bearing_angles : List, sizes : List, ownship_positions : List[NDArray]) -> WedgeEstimator:
    wedge = WedgeEstimator()
    wedge.set_velocity_position(bearing_angles, sizes, ownship_positions)
    return wedge

def binarize_matrix(matrix : NDArray, threshold : float) -> NDArray:
    matrix = np.array(matrix)
    binary_matrix = (matrix > threshold).astype(int)
    return binary_matrix


def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def get_neighbors(pos):
    directions = [
        # Directions with a cost of 1 (straight moves)
        ((0, 0, 1), 1), ((0, 0, -1), 1), ((0, 1, 0), 1), ((0, -1, 0), 1),
        ((1, 0, 0), 1), ((-1, 0, 0), 1),
        # Directions with a cost of √2 (diagonal moves in a plane)
        ((0, 1, 1), np.sqrt(2)), ((0, 1, -1), np.sqrt(2)), ((0, -1, 1), np.sqrt(2)), ((0, -1, -1), np.sqrt(2)),
        ((1, 1, 0), np.sqrt(2)), ((1, -1, 0), np.sqrt(2)), ((-1, 1, 0), np.sqrt(2)), ((-1, -1, 0), np.sqrt(2)),
        ((1, 0, 1), np.sqrt(2)), ((1, 0, -1), np.sqrt(2)), ((-1, 0, 1), np.sqrt(2)), ((-1, 0, -1), np.sqrt(2)),
        # Directions with a cost of √3 (3D diagonal moves)
        ((1, 1, 1), np.sqrt(3)), ((1, 1, -1), np.sqrt(3)), ((1, -1, 1), np.sqrt(3)), ((1, -1, -1), np.sqrt(3)),
        ((-1, 1, 1), np.sqrt(3)), ((-1, 1, -1), np.sqrt(3)), ((-1, -1, 1), np.sqrt(3)), ((-1, -1, -1), np.sqrt(3))
    ]
    return [(pos[0] + dx, pos[1] + dy, pos[2] + dz, cost) for (dx, dy, dz), cost in directions]


def bidirectional_a_star(grid, start, goal):
    if grid[start] == 1 or grid[goal] == 1:
        return None  # Start or goal is blocked

    def is_valid(pos):
        return (0 <= pos[0] < grid.shape[0] and
                0 <= pos[1] < grid.shape[1] and
                0 <= pos[2] < grid.shape[2] and
                grid[pos] == 0)

    def reconstruct_path(came_from, meeting_point):
        path = []
        current = meeting_point
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    open_start = []
    open_goal = []
    heapq.heappush(open_start, (0 + heuristic(start, goal), 0, start))
    heapq.heappush(open_goal, (0 + heuristic(goal, start), 0, goal))

    came_from_start = {start: None}
    came_from_goal = {goal: None}

    cost_so_far_start = {start: 0}
    cost_so_far_goal = {goal: 0}

    visited_start = set()
    visited_goal = set()

    while open_start and open_goal:
        # Expand from start
        _, current_cost_start, current_start = heapq.heappop(open_start)
        visited_start.add(current_start)

        if current_start in visited_goal:
            return reconstruct_path(came_from_start, current_start) + reconstruct_path(came_from_goal, current_start)[::-1][1:]

        for neighbor in get_neighbors(current_start):
            next_pos, move_cost = neighbor[:3], neighbor[3]
            if not is_valid(next_pos):
                continue
            new_cost = cost_so_far_start[current_start] + move_cost
            if next_pos not in cost_so_far_start or new_cost < cost_so_far_start[next_pos]:
                cost_so_far_start[next_pos] = new_cost
                priority = new_cost + heuristic(next_pos, goal)
                heapq.heappush(open_start, (priority, new_cost, next_pos))
                came_from_start[next_pos] = current_start

        # Expand from goal
        _, current_cost_goal, current_goal = heapq.heappop(open_goal)
        visited_goal.add(current_goal)

        if current_goal in visited_start:
            return reconstruct_path(came_from_start, current_goal) + reconstruct_path(came_from_goal, current_goal)[::-1][1:]

        for neighbor in get_neighbors(current_goal):
            next_pos, move_cost = neighbor[:3], neighbor[3]
            if not is_valid(next_pos):
                continue
            new_cost = cost_so_far_goal[current_goal] + move_cost
            if next_pos not in cost_so_far_goal or new_cost < cost_so_far_goal[next_pos]:
                cost_so_far_goal[next_pos] = new_cost
                priority = new_cost + heuristic(next_pos, start)
                heapq.heappush(open_goal, (priority, new_cost, next_pos))
                came_from_goal[next_pos] = current_goal

    return None


def get_in_out_wedges_and_vertices(wedges : List, plotting=False) -> Tuple[List[NDArray], List[NDArray]]:
    x, y = np.linspace(-params.solution_span, params.solution_span, 25), np.linspace(0, 2*params.solution_span, 25)
    X, Y = np.meshgrid(x, y)

    list_of_vertices = []
    in_out_wedge_list = []
    sim_time = 0
    sim_time += params.ts_simulation
    
    colors = ['r-', 'g-', 'y-']

    for i in range(25): 
        Z = np.zeros((25, 25))
        vertices = []
        points = np.vstack((Y.ravel(), X.ravel())).T
        idx_sec = params.measured_window + i*30
        for j, wedge in enumerate(wedges):
            vertice = wedge.get_wedge_vertices(sim_time)
            vertices.append(vertice)
            Z += are_inside_wedge(points, vertice).reshape(25, 25)

            if plotting:
                plt.plot([vertice[0,1], vertice[1,1], vertice[2,1], vertice[3,1], vertice[0,1]],
                         [vertice[0,0], vertice[1,0], vertice[2,0], vertice[3,0], vertice[0,0]], colors[j], linewidth=1, alpha=i/25)
        
        list_of_vertices.append(vertices)
        in_out_wedge_list.append(Z)
        sim_time += 30*params.ts_simulation
        if plotting:
            own_pos = wedges[0].init_own_pos + wedges[0].init_own_vel*sim_time
            plt.plot(own_pos[0], own_pos[1], 'ko', markersize=2, alpha=i/25)
            plt.xlim(-params.solution_span, params.solution_span)
            plt.ylim(0, 2*params.solution_span)
            plt.xlabel('E')
            plt.ylabel('N')
            plt.pause(0.01)
    if plotting:
        # plt.savefig('visualAvoidance2D/figures/alpha_wedge.png', dpi=300)
        plt.show()

    return in_out_wedge_list, list_of_vertices


def con_cltr_pnt(x : List, start_point : List) -> List:
    con = []
    # NOTE using the square root is probably not necessary but I cant get it to work without it. I square the bounded distance and it still won't work
    con = np.zeros(len(x)//2)
    x_temp = x.reshape(len(x)//2, 2).T
    diff = np.diff(x_temp)
    dist = np.linalg.norm(diff, axis=0)
    con = dist

    return con


def compute_cross_product2D(a : NDArray, b : NDArray) -> float:
    return a[0]*b[1] - a[1]*b[0]


def compute_relavant_cross_products(p : NDArray, vertices : NDArray) -> Tuple[float, float, float, float]:
    '''The vertices come in a list of 4 points, with the first point being the top right point of the wedge, 
       the second point being the bottom right point of the wedge, the third point being the bottom left point of the wedge,
       and the fourth point being the top left point of the wedge.'''
    right = compute_cross_product2D(vertices[0]-p, vertices[1]-p)
    bottom = compute_cross_product2D(vertices[1]-p, vertices[2]-p)
    left = compute_cross_product2D(vertices[2]-p, vertices[3]-p)
    top = compute_cross_product2D(vertices[3]-p, vertices[0]-p)
    return right, bottom, left, top


def project_vector_onto_vector(a : NDArray, b : NDArray) -> NDArray:
    return np.dot(a, b) / np.dot(b,b) * b


def compute_distance(a : NDArray, b : NDArray) -> float:
    return np.linalg.norm(a - b)


def compute_distance_to_edge(point : NDArray, v1 : NDArray, v2 : NDArray) -> float:
    v2_minus_point = point - v1
    v1_minus_v2 = v2 - v1
    projection = project_vector_onto_vector(v2_minus_point, v1_minus_v2)
    distance = compute_distance(v2_minus_point, projection)
    return distance


def get_distance_from_edge_with_point_inside_wedge(p : NDArray, vertices : NDArray) -> float:
    '''This function returns the distance from the point to the closest edge of the wedge'''
    distances = []
    distances.append(compute_distance_to_edge(p, vertices[0], vertices[1]))
    distances.append(compute_distance_to_edge(p, vertices[1], vertices[2]))
    distances.append(compute_distance_to_edge(p, vertices[2], vertices[3]))
    distances.append(compute_distance_to_edge(p, vertices[3], vertices[0]))
    return min(distances)


def distance_function(p : NDArray, vertices : NDArray) -> float:
    right, bottom, left, top = compute_relavant_cross_products(p, vertices)
    logic = np.array([right, bottom, left, top])
    if sum(logic < 0) == 4:
        return get_distance_from_edge_with_point_inside_wedge(p, vertices)
    
    if sum(logic > 0) == 2:
        if right >= 0 and bottom >= 0:
            return -compute_distance(p, vertices[1])
        elif bottom >= 0 and left >= 0:
            return -compute_distance(p, vertices[2])
        elif left >= 0 and top >= 0:
            return -compute_distance(p, vertices[3])
        elif top >= 0 and right >= 0:
            return -compute_distance(p, vertices[0])
        
    if sum(logic > 0) == 1:
        if right >= 0:
            return -compute_distance_to_edge(p, vertices[0], vertices[1])
        elif bottom >= 0:
            return -compute_distance_to_edge(p, vertices[1], vertices[2])
        elif left >= 0:
            return -compute_distance_to_edge(p, vertices[2], vertices[3])
        elif top >= 0:
            return -compute_distance_to_edge(p, vertices[3], vertices[0])
        
    if sum(logic > 0) == 3:
        return -compute_distance_to_edge(p, vertices[1], vertices[2])


def distance_constraint_vectorized(x : List[float], vertice_list : List[NDArray]) -> NDArray:
    num_intruders = len(vertice_list[0])
    distances = np.zeros(num_intruders*len(x)//2)

    for i in range(0, len(x), 2):
        for j, vertices in enumerate(vertice_list[i//2]):
            distances[num_intruders*i//2 + j] = distance_function(np.array([x[i], x[i+1]]), vertices)

    return distances


def object_function_new(x : List[float], goalPosition: Tuple[float, float]=(20., 20.)) -> float:
    '''Calculate the difference between the goal position and the current
    position of the distance'''
    res = 0
    res = (x[-2] - goalPosition[0])**2 + (x[-1] - goalPosition[1])**2
    return res


def initialize_x0(path : List[float], start : Tuple[float, float, float], end : Tuple[float, float, float], dim : int) -> Tuple[List[float], Tuple[float, float], Tuple[float, float]]:
    scaler_shift = 2*params.solution_span/d# print(distances)im
    x0 = []
    past = -1
    for i in range(len(path)):
        if path[i][0] != past:
            x0.append(scaler_shift*path[i][1])
            x0.append(scaler_shift*path[i][2] - params.solution_span)
        past = path[i][0]

    start_point = np.array([scaler_shift*start[1], scaler_shift*start[2]-params.solution_span])
    end_point = np.array([scaler_shift*end[1], scaler_shift*end[2]-params.solution_span])

    return x0, start_point, end_point


def get_optimal_control_points(x0 : List[NDArray], start_point : Tuple[float, float], end_point : Tuple[float, float], array_of_vertices : NDArray, safety_threshold : float = -100.) -> NDArray:
    vel_threshold = 2*params.solution_span/25.
    
    buffer = np.ones(2)*3
    start_point_constraint = LinearConstraint(np.eye(2, len(x0)), start_point-buffer, start_point+buffer)
    nlc = NonlinearConstraint(lambda x: con_cltr_pnt(x, start_point), 0.0, vel_threshold)
    dnlc = NonlinearConstraint(lambda x: distance_constraint_vectorized(x, array_of_vertices), -np.inf, safety_threshold)
    bounds = None
    res = minimize(object_function_new, x0, args=(np.array([end_point[0], end_point[1]]),), method='SLSQP', bounds=bounds, options={'maxiter':500, 'disp':True}, constraints=[nlc, dnlc, start_point_constraint], )
    return res


def get_knot_points(num_control_points : int, degree : int) -> NDArray:
    num_knots = num_control_points + degree + 1
    knots = np.concatenate((np.zeros(degree), np.linspace(0,1,num_knots-2*degree), np.ones(degree)))
    return knots


def get_bspline_path(control_points, degree : int) -> NDArray:
    knots = get_knot_points(len(control_points), degree)
    spl = BSpline(knots, control_points, degree)
    t = np.linspace(0, 1, 100)
    curve = spl(t)
    return curve