import heapq
from jax import jacfwd
import math
import scipy.stats as stats
from objective_function_ideas import *

def binarize_matrix(matrix, threshold):
    matrix = np.array(matrix)
    binary_matrix = (matrix > threshold).astype(int)
    # binary_matrix = (matrix > 0.5)
    return binary_matrix

def gauss_function_2d(x, x0, y0, sigma_x, sigma_y):
    '''Calculate the values of an unrotated Gauss function given positions
    in x and y in a mesh grid'''
    A = (1. / (2 * np.pi*sigma_x*sigma_y))
    return A*np.exp(-(x[0]-x0)**2/(2*sigma_x**2) -(x[1]-y0)**2/(2*sigma_y**2))

def inturder_linear_path(start, stop, num):
    '''Calculate the linear path between two points'''
    x = np.linspace(start[0], stop[0], num)
    y = np.linspace(start[1], stop[1], num)
    return x, y

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

def con_cltr_pnt(x, start_point):
    con = []
    # NOTE using the square root is probably not necessary but I cant get it to work without it. I square the bounded distance and it still won't work
    con = np.zeros(len(x)//2 - 1)
    x = np.array(list(x))
    # print(x)
    x_temp = x.reshape(-1, 2).T
    # print(x_temp)
    # con = []
    # dist = np.linalg.norm(x_temp[:,0] - start_point)
    # dist = np.sqrt((x[0]-start_point[0])**2 + (x[1]-start_point[1])**2)
    # print(dist)
    # con[0] = dist
    # con.append(dist)

    # diff = np.array([np.sqrt((x[i]-x[i+2])**2 + (x[i+1]-x[i+3])**2) for i in range(0, len(x)-2, 2)])
    diff = np.diff(x_temp)
    # print(diff)
    dist = np.linalg.norm(diff, axis=0)
    con = dist
    # for i in range(0, len(x)-2, 2):
    #     dist = (x[i]-x[i+2])**2 + (x[i+1]-x[i+3])**2
    #     con.append(dist)
    print(con)
    return con


def obj_crete(x, inturder_list):
    tmp = []

    for i in range(0, len(x), 2):
        out = 0
        for j in range(len(inturder_list)):
            # print(i, j)
            gau = gauss_function_2d([x[i], x[i+1]], inturder_list[j][0][int(i/2)], inturder_list[j][1][int(i/2)], 1.01**(i/2), 1.01**(i/2))
            out = out + gau
        tmp.append(out)
        # print(out)
    return tmp

def pdf_map(x, data):
    tmp = []
    # print(x.shape)
    # print(data.shape)
    for i in range(0, len(x), 2):
        # print(i, x[i], x[i+1])
        gmm = data[int(i/2) ,int(x[i]), int(x[i+1])]
        # print(gmm)
    tmp.append(gmm)
    # print(tmp)
    return tmp

def pdf_map_constraint(x, data):
    result = [] 
    for i in range(0, len(x), 2):
        # idx1 = int(np.round(x[i]))
        # idx2 = int(np.round(x[i+1]))
        
        idx1 = min(max(int(x[i]), 0), data.shape[1] - 1)
        idx2 = min(max(int(x[i+1]), 0), data.shape[2] - 1)
        result.append(data[int(i/2), idx1, idx2])
    return result

def pdf_map_constraint_functionized(x, pdf_functions):
    result = [] 
    
    for i in range(0, len(x), 2):
        idx1 = x[i]
        idx2 = x[i+1]
        value = pdf_functions[i//2](np.array([idx1, idx2]))
        result.append(value)
    return result

def pdf_map_constraint_functionized_fixed(x, wedges):
    result = [] 
    
    for i in range(0, len(x), 2):
        idx1 = x[i]
        idx2 = x[i+1]
        x_shifted = 1600*idx2 - 10000
        y_shifted = 800*idx1 - 5000
        value = sum(wedge.get_wedge_single_gaussian(4*i//2).pdf(np.array([y_shifted, x_shifted])) for wedge in wedges)
        # print(f't = {i/2}, idx1 = {idx1}, idx2 = {idx2}, x_shifted = {x_shifted}, y_shifted = {y_shifted}, value = {value}')
        result.append(value)
    return result

def pdf_map_constraint_functionized_shifted(x, wedges, size=25):
    result = [] 
    
    for i in range(0, len(x), 2):
        idx1 = x[i]
        idx2 = x[i+1]
        x_shifted = 20000/size*idx2 - 10000
        y_shifted = 20000/size*idx1 - 5000
        value = sum(wedge.get_wedge_single_gaussian(4*i//2).pdf(np.array([y_shifted, x_shifted])) for wedge in wedges)
        # print(f't = {i/2}, idx1 = {idx1}, idx2 = {idx2}, x_shifted = {x_shifted}, y_shifted = {y_shifted}, value = {value}')
        result.append(value)
    return result

def pdf_map_constraint_functionized_with_x0_preshifted(x, wedges):
    result = [] 
    
    for i in range(0, len(x), 2):
        idx1 = x[i]
        idx2 = x[i+1]
        value = sum(wedge.get_wedge_single_gaussian(4*i//2).pdf(np.array([idx1, idx2])) for wedge in wedges)
        result.append(value)
    return result

def pdf_map_constraint_list_with_x0_preshifted(x, pdf_functions):
    result = [] 
    
    for i in range(0, len(x), 2):
        idx1 = x[i]
        idx2 = x[i+1]
        value = pdf_functions[i//2](np.array([idx1, idx2]))
        result.append(value)
    return result

def pdf_map_constraint_functionized_list(x, pdf_functions):
    result = []
    
    for i in range(0, len(x), 2):
        idx1 = x[i]
        idx2 = x[i+1]
        x_shifted = 1600*idx2 - 20000
        y_shifted = 800*idx1 - 5000
        result.append(pdf_functions[4*i//2](np.array([y_shifted, x_shifted])))
    
    return result


def object_function(x, goalPosition=(20,20)):
    '''Calculate the difference between the goal position and the current
    position of the distance'''
    res = 0
    for i in range(0, len(x), 2):
        tmp = math.dist((x[i], x[i+1]), goalPosition)
        res = tmp + res
    # print(res)
    return res

def object_function_new(x, goalPosition=(20,20)):
    '''Calculate the difference between the goal position and the current
    position of the distance'''
    res = 0
    res = (x[-2] - goalPosition[0])**2 + (x[-1] - goalPosition[1])**2
    return res




def inside_wedge(x, vertices):
    '''Check if a point is inside a single wedge'''
    def sign(a, b, o):
            # get the determinate of [a-o, b-o]
            return (a[0] - o[0]) * (b[1] - o[1]) - (b[0] - o[0]) * (a[1] - o[1])
    
    v1, v2, v3, v4 = vertices
    d1 = sign(v1, v2, x)
    d2 = sign(v2, v3, x)
    d3 = sign(v3, v4, x)
    d4 = sign(v4, v1, x)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)
    
    return float(not (has_neg and has_pos))

def inside_all_wedges(x, vertices_list):
    '''Check if a point is inside all the wedges'''
    for vertices in vertices_list:
        if inside_wedge(x, vertices) == 1.0:
            return 1.0
    return 0.0