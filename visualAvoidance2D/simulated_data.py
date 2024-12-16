import numpy as np
import matplotlib.pyplot as plt
from utilities import WedgeEstimator, are_inside_wedge

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

def create_wedges(ownship, intruders, plot=False, button_press=False):
    num_intruders = len(intruders)
    bearings = []
    sizes = []
    rhos = []
    for i in range(num_intruders):
        bearings.append([])
        sizes.append([])
        rhos.append([])
        for intruder_pos, ownship_pos in zip(intruders[i], ownship):
            bearing = np.arctan2(intruder_pos[0] - ownship_pos[0], intruder_pos[1] - ownship_pos[1])
            rho = np.linalg.norm(intruder_pos - ownship_pos)
            size = intruder_wingspan / rho
            bearings[i].append(bearing)
            sizes[i].append(size)
            rhos[i].append(rho)

    wedges = []
    num = 10
    for i in range(num_intruders):
        wedge = WedgeEstimator()
        wedge.set_velocity_position(bearings[i][:num], sizes[i][:num], [0.0]*num, ownship[:num].reshape(num,2,1))
        wedges.append(wedge)

    t = 0
    t += ts

    if plot:
        for i, intruder in enumerate(intruders):
            plt.plot(intruder[:,0], intruder[:,1], 'ro', label=f"Intruder {i+1}")
        plt.plot(ownship[:,0], ownship[:,1], 'bo', label='Ownship')
        plt.show()

        # plot_bearings_sizes_rhos(bearings, sizes, rhos)

        while num < len(ownship):
            plt.clf()

            own_pos = wedge.init_own_pos
            own_vel = wedge.init_own_vel
            pose = own_pos + own_vel * t
            plt.plot(ownship[num,0], ownship[num,1], 'bo', linewidth=1)
            for i, intruder in enumerate(intruders):
                vertices = wedges[i].get_wedge_vertices(t)
                plt.plot(intruder[num,0], intruder[num,1], 'ro', linewidth=1)
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
    x, y = np.linspace(start[0] - 5000, start[0] + 5000, 25), np.linspace(start[1] - 1000, start[1] + 9000, 25)
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
        list_of_vertices.append(vertices)
        in_out_wedge_list.append(Z)

    return in_out_wedge_list, list_of_vertices





if __name__ == '__main__':
    filepath = 'visualAvoidance2D/data/xplane_data/0004/20241205_152650_all_positions_in_path.npy'
    ownship, intruders = get_ownship_intruder_positions(filepath)
    create_wedges(ownship, intruders, plot=True, button_press=False)