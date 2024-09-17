import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from pathplannerutility import *
import math
from scipy.optimize import minimize, NonlinearConstraint, differential_evolution
import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure, show
# from scipy.optimize import minimize, NonlinearConstraint
# from matplotlib import cm
# from matplotlib.colors import LinearSegmentedColormap
# import heapq
# from skimage.filters import threshold_otsu
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import BSpline
# get colormap
ncolors = 256
color_array = plt.get_cmap('jet')(range(ncolors))

# change alpha values
color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)

# register this new colormap with matplotlib
plt.colormaps.register(cmap=map_object)



data = np.load('visualAvoidance2D/data/gaussian_map3.npy')
ownship = np.load('visualAvoidance2D/data/ownship_positions2.npy')

original_shape = data.shape
print("Original shape:", original_shape)

scale_resolution = 1
probability_threshold = 0.00000001
new_shape = (25, 25, 25)
reshaped_data = data.reshape(new_shape[0], original_shape[0]//new_shape[0],
                             new_shape[1], original_shape[1]//new_shape[1],
                             new_shape[2], original_shape[2]//new_shape[2])

downsampled_data = reshaped_data.mean(axis=(1, 3, 5))
data = downsampled_data

print("Downsampled shape:", downsampled_data.shape)
start = (0, int(ownship[25][0]/2500), int((ownship[25][1]+5000)/2500)) #time North East

# start = (0, 0, 56)
start = (0, 0, 14)
# goal = (99, 99, 65)
goal = (24, 24, 15)
print("start point:",start, "goal point:", goal)

binary_matrix = binarize_matrix(data, 0.00000001)
path = bidirectional_a_star(binary_matrix, start, goal)
# print(binary_matrix)
print("path:", path)
print("path length:", len(path))

int_X0 = []
past = -1
for i in range(0, len(path)):
    # print(i,path[i])
    if path[i][0] != past:
        int_X0.append(path[i][1]*scale_resolution)
        int_X0.append(path[i][2]*scale_resolution)
    past = path[i][0]

print("int_X0:", len(int_X0))
start_point = [start[1], start[2]]
print("start_point:", start_point)
goal_point = [goal[1], goal[2]]
print("goal_point:", goal_point)

nlc = NonlinearConstraint(lambda x: con_cltr_pnt(x, start_point), 0.0, 1.0)
P_nlc = NonlinearConstraint(lambda x: pdf_map_constraint(x, binary_matrix), 0.0, probability_threshold)
res = minimize(object_function, int_X0, args=((goal_point[0], goal_point[1]),), method='SLSQP', bounds=[(-1, 26) for i in range(len(int_X0))], constraints=[nlc, P_nlc])

print(res.success)
print(res.message)
print(len(res.x))
# print(res.x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Show the plot
z = np.arange(data.shape[0])
y = np.arange(data.shape[1])
x = np.arange(data.shape[2])

x, y = np.meshgrid(x, y)
voxels_transposed = np.transpose(binary_matrix, (2, 1, 0))
ax.voxels(voxels_transposed, edgecolor='none', alpha=0.1)

for i in range(0, data.shape[0]):
   sc =  ax.contourf(x, y, data[i, :, :], 100, zdir='z', offset=i, cmap='rainbow_alpha')
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Color Scale')


# Plot the path
z_coords = [point[0] for point in path]
x_coords = [point[2] for point in path]
y_coords = [point[1] for point in path]
ax.plot(x_coords, y_coords, z_coords, label='Path', color='green', linewidth=3, zorder=1)

# Plot the optimal path
# for i in range(0, len(res.x), 2):
#     ax.scatter3D(res.x[i+1], res.x[i], int(i/2))
#     # print(res.x[i], res.x[i+1])
#     ax.annotate(int(i/2), (res.x[i], res.x[i+1]), color='red')
#     ax.text(res.x[i+1], res.x[i], int(i/2), '%s' % int(i/2), size=10, zorder=1, color='k')

for i in range(0, len(int_X0), 2):
    ax.scatter3D(int_X0[i+1], int_X0[i], int(i/2))
    # print(int_X0[i], int_X0[i+1])
    # ax.annotate(int(i/2), (int_X0[i], int_X0[i+1]), color='red')
    # ax.text(int_X0[i+1], int_X0[i], int(i/2), '%s' % int(i/2), size=10, zorder=1, color='k')

# # Convert int_X0 to numpy array
# control_points = np.array(int_X0).reshape(-1, 2)
# # Add the first and last control points to ensure the start and goal points are included
# control_points = np.vstack([start_point, control_points, goal_point])
# # Create BSpline object
# t = np.arange(len(control_points))
# k = 2  # Degree of the spline
# spl = BSpline(t, control_points, k)
# # Evaluate the spline
# t_new = np.linspace(0, len(control_points)-1, 100)  # New parameter values
# spline_points = spl(t_new)
# # Plot the spline
# ax.plot(spline_points[:, 1], spline_points[:, 0], t_new, label='Spline', color='blue', linewidth=3, zorder=1)


ax.set_zlim(0, 25)
ax.set_ylim(0, 25)
ax.set_xlim(0, 25)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Time axis')
# plt.savefig("hree-intruder scenario.svg")
plt.show()
# import plotly.graph_objects as go



# fig = go.Figure(data=[
#     go.Scatter3d(
#         x=x_coords,
#         y=y_coords,
#         z=z_coords,
#         mode='lines',
#         line=dict(color='green', width=3),
#         name='Path'
#     ),
#     go.Scatter3d(
#         x=res.x[1::2],
#         y=res.x[::2],
#         z=list(range(len(res.x)//2)),
#         mode='markers',
#         marker=dict(color='red', size=5),
#         name='Optimal Path'
#     )
# ])
# fig = go.Figure()
# fig.add_trace(go.Scatter3d(
#         x=x_coords,
#         y=y_coords,
#         z=z_coords,
#         mode='lines',
#         line=dict(color='green', width=3),
#         name='Path'
# ))

# fig.add_trace(go.Scatter3d(
#     x=res.x[1::2],
#     y=res.x[::2],
#     z=list(range(len(res.x)//2)),
#     mode='markers',
#     marker=dict(color='red', size=5),
#     name='Optimal Path'
# ))

# fig.add_trace(go.Volume(
#     x=x.flatten(),
#     y=y.flatten(),
#     z=z.flatten(),
#     value=binary_matrix.flatten(),
#     isomin=0,
#     isomax=1,
#     opacity=1.0,
#     surface_count=50,
#     colorscale='Viridis',
#     showscale=False,
#     name='Voxel'
# ))

# fig.update_layout(
#     scene=dict(
#         xaxis_title='X axis',
#         yaxis_title='Y axis',
#         zaxis_title='Time axis',
#         xaxis=dict(range=[0, data.shape[1]]),
#         yaxis=dict(range=[0, data.shape[2]]),
#         zaxis=dict(range=[0, data.shape[0]])
#     )
# )


# fig.show()