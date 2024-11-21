from objective_function_ideas import *
import numpy as np
import matplotlib.pyplot as plt

def test1():
    a = np.array([0, 2])
    b = np.array([2, 0])
    print(compute_distance(a, b))
    print(project_vector_onto_vector(a, b))
    print(compute_distance_to_edge(np.array([0,0]), a, b) == np.linalg.norm(np.array([1,1])))
    temp_vec = a-project_vector_onto_vector(a, b)
    print(np.linalg.norm(temp_vec))
    plt.plot([0,a[0]], [0,a[1]], 'r-')
    plt.plot([0,b[0]], [0,b[1]], 'b-')
    plt.plot([0,temp_vec[0]], [0,temp_vec[1]], 'g-')
    plt.show()

def test_helper_function(p, vertices):
    print('Testing point ', p)
    print('Inside wedge ', is_inside_wedge(p, vertices))
    print('In e1 range ', is_in_e1(p, vertices))
    print('In e2 range ', is_in_e2(p, vertices))
    print('In e3 range ', is_in_e3(p, vertices))
    print('In e4 range ', is_in_e4(p, vertices))
    print('In v1 range ', is_in_v1(p, vertices))
    print('In v2 range ', is_in_v2(p, vertices))
    print('In v3 range ', is_in_v3(p, vertices))
    print('In v4 range ', is_in_v4(p, vertices), '\n')

def test_inside_wedge():
    vertices = np.array([[3,3],[3,1],[1,1],[1,3]])
    p = np.array([2,2])
    test_helper_function(p, vertices)

def test_in_e1_range():
    vertices = np.array([[3,3],[3,1],[1,1],[1,3]])
    p = np.array([4,2])
    test_helper_function(p, vertices)

def test_in_e2_range():
    vertices = np.array([[3,3],[3,1],[1,1],[1,3]])
    p = np.array([2,0])
    test_helper_function(p, vertices)

def test_in_e3_range():
    vertices = np.array([[3,3],[3,1],[1,1],[1,3]])
    p = np.array([0,2])
    test_helper_function(p, vertices)

def test_in_e4_range():
    vertices = np.array([[3,3],[3,1],[1,1],[1,3]])
    p = np.array([2,4])
    test_helper_function(p, vertices)

def test_in_v1_range():
    vertices = np.array([[3,3],[3,1],[1,1],[1,3]])
    p = np.array([4,4])
    test_helper_function(p, vertices)

def test_in_v2_range():
    vertices = np.array([[3,3],[3,1],[1,1],[1,3]])
    p = np.array([4,0])
    test_helper_function(p, vertices)

def test_in_v3_range():
    vertices = np.array([[3,3],[3,1],[1,1],[1,3]])
    p = np.array([0,0])
    test_helper_function(p, vertices)

def test_in_v4_range():
    vertices = np.array([[3,3],[3,1],[1,1],[1,3]])
    p = np.array([0,4])
    test_helper_function(p, vertices)

def test_distance_function():
    n = 100
    vertices = np.array([[n/2,n/2],[n/2, -n/2],[-n/2, -n/2],[-n/2,n/2]])
    vertices2 = vertices + 50
    # print(vertices.shape)
    x, y = np.linspace(-2*n,2*n,100), np.linspace(-2*n,2*n,100)
    X, Y = np.meshgrid(x, y)
    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z1[i,j] = distance_function(np.array([X[i,j], Y[i,j]]), vertices)
            Z2[i,j] = distance_function(np.array([X[i,j], Y[i,j]]), vertices2)
    Z = Z1 + Z2
    print(np.min(Z))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([vertices[0,0], vertices[1,0], vertices[2,0], vertices[3,0], vertices[0,0]], 
            [vertices[0,1], vertices[1,1], vertices[2,1], vertices[3,1], vertices[0,1]], 'hotpink', linewidth=2)
    ax.plot([vertices2[0,0], vertices2[1,0], vertices2[2,0], vertices2[3,0], vertices2[0,0]],
            [vertices2[0,1], vertices2[1,1], vertices2[2,1], vertices2[3,1], vertices2[0,1]], 'hotpink', linewidth=2)
    ax.plot_surface(X, Y, Z, cmap='viridis')
    # ax.set_zlim([-1, 10])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('distance')

    plt.show()

def test_vertices():
    list_of_vertices = np.load('visualAvoidance2D/data/vertices.npy')
    idx = np.random.choice(len(list_of_vertices), 3)
    vertices_of_interest = list_of_vertices[idx]
    
    for vertices in vertices_of_interest:
        vertices = vertices.reshape(4,2).copy()
        minx, miny = np.min(vertices, axis=0)
        maxx, maxy = np.max(vertices, axis=0)
        x, y = np.linspace(minx-100, maxx+100, 200), np.linspace(miny-100, maxy+100, 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i,j] = np.exp(distance_function(np.array([X[i,j], Y[i,j]]), vertices)/500) - 1
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([vertices[0,0], vertices[1,0], vertices[2,0], vertices[3,0], vertices[0,0]], 
                [vertices[0,1], vertices[1,1], vertices[2,1], vertices[3,1], vertices[0,1]], 'hotpink', linewidth=2)
        
        ax.plot_surface(X, Y, Z, cmap='viridis')
        # add color bar
        # cbar = plt.colorbar(ax.plot_surface(X, Y, np.exp(Z) - 1, cmap='viridis'))

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('distance')

        plt.show()
    
def test_compute_relavant_cross_products_vectorized():
    points = np.array([[0,0], [1,1], [2,2], [3,3]])
    vertices = np.array([[3,3],[3,1],[1,1],[1,3]])
    print(compute_relevant_cross_products_vectorized(points, vertices))

def test_is_inside_wedge_vectorized():
    x, y = np.linspace(0, 5, 100), np.linspace(0, 5, 100)
    X, Y = np.meshgrid(x, y)
    points = np.stack((X.flatten(), Y.flatten()), axis=1)
    vertices = np.array([[3,3],[3,1],[1,1],[1,3]])
    is_in = is_inside_wedge_vectorized(points, vertices)
    Z = is_in.reshape(X.shape)
    plt.contourf(X, Y, Z, cmap='viridis')
    plt.show()

if __name__ == '__main__':
    # test1()
    # test_inside_wedge()
    # test_in_e1_range()
    # test_in_e2_range()
    # test_in_e3_range()
    # test_in_e4_range()
    # test_in_v1_range()
    # test_in_v2_range()
    # test_in_v3_range()
    # test_in_v4_range()
    # print('All tests passed')
    test_distance_function()
    # test_vertices()
    # test_compute_relavant_cross_products_vectorized()
    # test_is_inside_wedge_vectorized()