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


if __name__ == '__main__':
    test1()
    # test_inside_wedge()
    # test_in_e1_range()
    # test_in_e2_range()
    # test_in_e3_range()
    # test_in_e4_range()
    # test_in_v1_range()
    # test_in_v2_range()
    # test_in_v3_range()
    # test_in_v4_range()
    print('All tests passed')