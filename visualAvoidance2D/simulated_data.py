import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from scipy.optimize import NonlinearConstraint, minimize

from utilities import WedgeEstimator, are_inside_wedge
from pathplannerutility import bidirectional_a_star, binarize_matrix, con_cltr_pnt, object_function_new
from objective_function_ideas import distance_constraint
from plotting_bspline_trajectory import get_bspline_path

intruder_wingspan = 20
ts = 1/30

def get_ownship_intruder_positions(filepath):
    real = np.load(filepath)
    ownship = real[0]
    intruders = []
    num_intruders = real.shape[0] - 1
    for intruder in real[1:]:
        intruders.append(intruder)
    return ownship, intruders

def get_bearing_size_measurements(filepath):
    data = np.load(filepath)
    bearings = [[] for _ in range(data.shape[1])]
    sizes = [[] for _ in range(data.shape[1])]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            bearings[j].append(data[i][j,0])
            sizes[j].append(data[i][j,1])
    return bearings, sizes

def create_wedges(ownship, intruders, bearing_measurements, pixel_size, plot=False, button_press=False):
    num_intruders = len(intruders)
    bearings = []
    sizes = []
    rhos = []
    for i in range(num_intruders):
        bearings.append([])
        sizes.append([])
        rhos.append([])
        for intruder_pos, ownship_pos, bearing, size in zip(intruders[i], ownship, bearing_measurements[i], pixel_size[i]):
            rho = np.linalg.norm(intruder_pos - ownship_pos)
            size = intruder_wingspan / rho
            bearings[i].append(bearing)
            sizes[i].append(size)
            rhos[i].append(rho)

    wedges = []
    num = 30
    for i in range(num_intruders):
        wedge = WedgeEstimator()
        wedge.set_velocity_position(bearings[i][:num], sizes[i][:num], [0.0]*num, ownship[:num].reshape(num,2,1))
        wedges.append(wedge)

    t = 0
    t += ts

    if plot:
        colors = ['ro', 'go', 'yo']
        for j in range(len(ownship)):
            # plt.clf()
            for i, intruder in enumerate(intruders):
                plt.plot(intruder[j,0], intruder[j,1], colors[i], label=f"Intruder {i+1}")
            plt.plot(ownship[j,0], ownship[j,1], 'ko', label='Ownship')
            plt.xlim(-300, 1000)
            plt.ylim(-100, 1300)
            if button_press:
                plt.waitforbuttonpress()
            else:
                plt.pause(0.01)

        plt.legend()            
        plt.show()

        plot_bearings_sizes_rhos(bearings, sizes, rhos)

        while num < len(ownship):
            plt.clf()

            own_pos = wedge.init_own_pos
            own_vel = wedge.init_own_vel
            pose = own_pos + own_vel * t
            plt.plot(ownship[num,0], ownship[num,1], 'bo', linewidth=1)
            for i, intruder in enumerate(intruders):
                vertices = wedges[i].get_wedge_vertices(t)
                plt.plot(intruder[num,0], intruder[num,1], 'go', markersize=2)
                mid_top = (vertices[0] + vertices[3]) / 2
                mid_bottom = (vertices[1] + vertices[2]) / 2
                plt.plot([vertices[0,1], vertices[1,1], vertices[2,1], vertices[3,1], vertices[0,1]],
                        [vertices[0,0], vertices[1,0], vertices[2,0], vertices[3,0], vertices[0,0]], 'r', linewidth=1)
                plt.plot([pose[0,0], mid_top[1,0]], [pose[1,0], mid_top[0,0]], 'k', linewidth=1)
            t += ts
            plt.xlim(-300, 1000)
            plt.ylim(-100, 1300)
            num += 1
            if button_press:
                plt.waitforbuttonpress()
            else:
                plt.pause(0.01)

        plt.show()

    # print(vertices)

    return wedges

def plot_bearings_sizes_rhos(bearings, sizes, rhos):
    for i in range(len(bearings)):
        plt.subplot(3,1,1)
        plt.plot(bearings[i], 'o')
        plt.title('Bearings')

        plt.subplot(3,1,2)
        plt.plot(sizes[i], 'o')
        plt.title('Sizes')

        plt.subplot(3,1,3)
        plt.plot(rhos[i], 'o')
        plt.title('Rhos')
        plt.tight_layout()
        plt.show()

def make_voxel_map_for_a_star(wedges, ownship):
    start = ownship[11]
    x, y = np.linspace(start[0] - 2500, start[0] + 2500, 25), np.linspace(start[1], start[1] + 5000, 25)
    X, Y = np.meshgrid(x, y)

    list_of_vertices = []
    in_out_wedge_list = []
    sim_time = 0
    sim_time += ts

    for i in range(25): 
        Z = np.zeros((25, 25))
        vertices = []
        points = np.vstack((Y.ravel(), X.ravel())).T
        for wedge in wedges:
            vertice = wedge.get_wedge_vertices(sim_time)
            vertices.append(vertice)
            Z += are_inside_wedge(points, vertice).reshape(25, 25)
            plt.plot([vertice[0,1], vertice[1,1], vertice[2,1], vertice[3,1], vertice[0,1]],
                        [vertice[0,0], vertice[1,0], vertice[2,0], vertice[3,0], vertice[0,0]], 'r', linewidth=1)
        list_of_vertices.append(vertices)
        in_out_wedge_list.append(Z)
        sim_time += 30*ts
        plt.xlim(start[0]-2500, start[0] + 2500)
        plt.ylim(start[1], start[1] + 4000)
        plt.pause(0.01)
    plt.show()

    return in_out_wedge_list, list_of_vertices

def initialize_x0(path, start, end, dim, ownship_start):
    scaler_shift = 5000/dim
    x0 = []
    past = -1
    for i in range(len(path)):
        if path[i][0] != past:
            x0.append(scaler_shift*path[i][1] + ownship_start[1])
            x0.append(scaler_shift*path[i][2] - 2500 + ownship_start[0])
        past = path[i][0]

    start_point = np.array([scaler_shift*start[1] + ownship_start[1], scaler_shift*start[2]-2500 + ownship_start[0]])
    end_point = np.array([scaler_shift*end[1] + ownship_start[1], scaler_shift*end[2]-2500 + ownship_start[0]])

    return x0, start_point, end_point
    
def optimize_path(x0, start_point, end_point, array_of_vertices):
    nlc = NonlinearConstraint(lambda x: con_cltr_pnt(x, start_point), 0.0, 200.0)
    dnlc = NonlinearConstraint(lambda x: distance_constraint(x, array_of_vertices), -np.inf, -50*num_intruders)
    bounds = None
    res = minimize(object_function_new, x0, args=(np.array([end_point[0], end_point[1]]),), method='SLSQP', bounds=bounds, options={'maxiter':500, 'disp':True}, constraints=[nlc, dnlc], )
    return res
    

def plot_solution(x0, res, list_of_vertices, ownship_start):
    x, y = np.linspace(ownship_start[0] - 2500, ownship_start[0] + 2500, 25), np.linspace(ownship_start[1], ownship_start[1] + 5000, 25)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, vertices in enumerate(list_of_vertices):
        for vertice in vertices:
            ax.plot([vertice[0,1], vertice[1,1], vertice[2,1], vertice[3,1], vertice[0,1]],
                    [vertice[0,0], vertice[1,0], vertice[2,0], vertice[3,0], vertice[0,0]], i, 'r', linewidth=1)

    for i in range(0, len(res.x), 2):
        ax.plot(res.x[i+1], res.x[i], int(i/2), 'o', color='blue', markersize=4)

    ax.plot(res.x[-1], res.x[-2], 24, 'o', color='blue', markersize=4, label='Optimal control points')

    ax.set_zlim(0, 25)
    ax.set_xlim([ownship_start[0] - 2500, ownship_start[0] + 2500])
    ax.set_ylim([ownship_start[1], ownship_start[1] + 5000])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Time axis')
    ax.legend()
    plt.show()

def animate_path(ownship_start, curve, list_of_vertices):
    fig, ax = plt.subplots()
    
    def update(frame):
        ax.cla()

        time_with_frame = frame*4
        print(time_with_frame, frame)
        ax.plot(curve[:,1], curve[:,0], 'g-')
        ax.plot(curve[time_with_frame,1], curve[time_with_frame,0], 'ro')

        for vertice in list_of_vertices[frame]:
            ax.plot([vertice[0,1], vertice[1,1], vertice[2,1], vertice[3,1], vertice[0,1]],
                    [vertice[0,0], vertice[1,0], vertice[2,0], vertice[3,0], vertice[0,0]], 'r', linewidth=1)
            
        ax.set_xlim([ownship_start[0] - 2500, ownship_start[0] + 2500])
        ax.set_ylim([ownship_start[1], ownship_start[1] + 5000])
    
    ani = animation.FuncAnimation(fig, update, frames=range(len(list_of_vertices)), repeat=False)

    plt.show()
    


if __name__ == '__main__':
    filepath_real = 'visualAvoidance2D/data/xplane_data/0004/20241205_152650_all_positions_in_path.npy'
    filepath_bearing = 'visualAvoidance2D/data/xplane_data/0004/20241205_152650_bearing_info.npy'

    ownship, intruders = get_ownship_intruder_positions(filepath_real)
    bearings, sizes = get_bearing_size_measurements(filepath_bearing)
    num_intruders = len(intruders)

    wedges = create_wedges(ownship, intruders, bearings, sizes, plot=True, button_press=False)
    in_out_wedge_list, list_of_vertices = make_voxel_map_for_a_star(wedges, ownship)
    data = np.array(in_out_wedge_list)
    binary_matrix = binarize_matrix(data, 0)
    start = (0,0,13)
    end = (24,24,13)
    
    path = bidirectional_a_star(binary_matrix, start, end)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    voxels_transposed = np.transpose(data, (2, 1, 0))
    ax.voxels(voxels_transposed, edgecolor='none', alpha=0.5)

    for i in range(0, len(path)):
        ax.plot(path[i][2], path[i][1], path[i][0], 'o', color='blue', markersize=4)

    plt.show()

    x0, start_point, end_point = initialize_x0(path, start, end, 25, ownship[11])
    array_of_vertices = np.array(list_of_vertices).reshape(25,num_intruders,4,2)
    print(start, end)
    print(start_point, end_point)
    start = time.time()
    res = optimize_path(x0, start_point, end_point, array_of_vertices)
    print("Optimization time: ", round(time.time() - start,5), " seconds")

    curve = get_bspline_path(res.x.reshape(-1,2), 3)
    plot_solution(x0, res, list_of_vertices, ownship[11])
    animate_path(ownship[11], curve, list_of_vertices)





