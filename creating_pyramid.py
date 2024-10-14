import numpy as np
import matplotlib.pyplot as plt

def get_middle_point(vertices):
    middle_bottom = (vertices[1] + vertices[2]) / 2
    middle_top = (vertices[0] + vertices[3]) / 2
    intruder_dir = middle_top - middle_bottom
    middle_left = (vertices[2] + vertices[3]) / 2
    middle_right = (vertices[0] + vertices[1]) / 2
    perp_direction = middle_right - middle_left
    perp_dist = np.linalg.norm(perp_direction)
    middle = middle_bottom + 0.5*intruder_dir
    return middle

def plot_pyramid(vertices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate the middle point and set it as the apex of the pyramid
    middle = get_middle_point(vertices)
    apex = np.array([middle.item(0), middle.item(1), 1]) # Elevate the apex to z=1
    
    # Plot the base of the pyramid on the XY plane (z=0)
    for i in range(4):
        next_i = (i + 1) % 4
        ax.plot([vertices[i, 0], vertices[next_i, 0]], 
                [vertices[i, 1], vertices[next_i, 1]], 
                [0, 0], 'r-')
    
    # # Connect each vertex of the base to the apex
    for i in range(4):
        ax.plot([vertices[i].item(0), apex[0]], 
                [vertices[i].item(1), apex[1]], 
                [0, apex[2]], 'r-')
        
    plt.show()

vertices = np.load('list_of_vertices.npy')
vertices = vertices[0]
print(vertices[0][0])

plot_pyramid(vertices)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot((2, 1), (2, 1), (0, 1))
ax.plot((0,1),(0,1),(0,1))
ax.plot((0,1),(2,1),(0,1))
ax.plot((2,1),(0,1),(0,1))

plt.show()