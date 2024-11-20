import jax.numpy as np
import jax
import matplotlib.pyplot as plt
import numpy

# Enable 64-bit precision globally
jax.config.update("jax_enable_x64", True)


def compute_cross_product2D(a, b):
    return a[0]*b[1] - a[1]*b[0]

def compute_distance(a, b):
    return np.linalg.norm(a - b)

def project_vector_onto_vector(a, b):
    return np.dot(a, b) / np.dot(b,b) * b

def compute_distance_to_edge(point, v1, v2):
    v2_minus_point = point - v1
    v1_minus_v2 = v2 - v1
    projection = project_vector_onto_vector(v2_minus_point, v1_minus_v2)
    distance = compute_distance(v2_minus_point, projection)
    return distance

def compute_relavant_cross_products(p, vertices):
    '''The vertices come in a list of 4 points, with the first point being the top right point of the wedge, 
       the second point being the bottom right point of the wedge, the third point being the bottom left point of the wedge,
       and the fourth point being the top left point of the wedge.'''
    right = compute_cross_product2D(vertices[0]-p, vertices[1]-p)
    bottom = compute_cross_product2D(vertices[1]-p, vertices[2]-p)
    left = compute_cross_product2D(vertices[2]-p, vertices[3]-p)
    top = compute_cross_product2D(vertices[3]-p, vertices[0]-p)
    return right, bottom, left, top

def is_inside_wedge(p, vertices):
    '''The vertices come in a list of 4 points, with the first point being the top right point of the wedge, 
       the second point being the bottom right point of the wedge, the third point being the bottom left point of the wedge,
       and the fourth point being the top left point of the wedge.'''

    right, bottom, left, top = compute_relavant_cross_products(p, vertices)
    if right < 0 and bottom < 0 and left < 0 and top < 0:
        return True
    else:
        return False
    
def is_in_e1(p, vertices):
    '''This function checks if the point is on the right side of edge 1. That being v2 to v1'''
    right, bottom, left, top = compute_relavant_cross_products(p, vertices)
    if right > 0 and bottom < 0 and left < 0 and top < 0:
        return True
    else:
        return False
    
def is_in_e2(p, vertices):
    '''This function checks if the point is on the right side of edge 2. That being v3 to v2'''
    right, bottom, left, top = compute_relavant_cross_products(p, vertices)
    if right < 0 and bottom > 0 and left < 0 and top < 0:
        return True
    else:
        return False
    
def is_in_e3(p, vertices):
    '''This function checks if the point is on the right side of edge 3. That being v4 to v3'''
    right, bottom, left, top = compute_relavant_cross_products(p, vertices)
    if right < 0 and bottom < 0 and left > 0 and top < 0:
        return True
    else:
        return False

def is_in_e4(p, vertices):
    '''This function checks if the point is on the right side of edge 4. That being v1 to v4'''
    right, bottom, left, top = compute_relavant_cross_products(p, vertices)
    if right < 0 and bottom < 0 and left < 0 and top > 0:
        return True
    else:
        return False
    
def is_in_v1(p, vertices):
    '''This function checks if the point is in the v1 range'''
    right, bottom, left, top = compute_relavant_cross_products(p, vertices)
    if right >= 0 and bottom < 0 and left < 0 and top >= 0:
        return True
    else:
        return False

def is_in_v2(p, vertices):
    '''This function checks if the point is in the v2 range'''
    right, bottom, left, top = compute_relavant_cross_products(p, vertices)
    if right >= 0 and bottom >= 0 and left < 0 and top < 0:
        return True
    else:
        return False

def is_in_v3(p, vertices):
    '''This function checks if the point is in the v3 range'''
    right, bottom, left, top = compute_relavant_cross_products(p, vertices)
    if right < 0 and bottom >= 0 and left >= 0 and top < 0:
        return True
    else:
        return False

def is_in_v4(p, vertices):
    '''This function checks if the point is in the v4 range'''
    right, bottom, left, top = compute_relavant_cross_products(p, vertices)
    if right < 0 and bottom < 0 and left >= 0 and top >= 0:
        return True
    else:
        return False


def get_distance_from_edge_with_point_inside_wedge(p, vertices):
    '''This function returns the distance from the point to the closest edge of the wedge'''
    distances = []
    distances.append(compute_distance_to_edge(p, vertices[0], vertices[1]))
    distances.append(compute_distance_to_edge(p, vertices[1], vertices[2]))
    distances.append(compute_distance_to_edge(p, vertices[2], vertices[3]))
    distances.append(compute_distance_to_edge(p, vertices[3], vertices[0]))
    return min(distances)

def distance_function(p, vertices):
    if is_inside_wedge(p, vertices):
        return get_distance_from_edge_with_point_inside_wedge(p, vertices)
    
    if is_in_e1(p, vertices):
        return -compute_distance_to_edge(p, vertices[0], vertices[1])
    
    if is_in_e2(p, vertices):
        return -compute_distance_to_edge(p, vertices[1], vertices[2])
    
    if is_in_e3(p, vertices):
        return -compute_distance_to_edge(p, vertices[2], vertices[3])
    
    if is_in_e4(p, vertices):
        return -compute_distance_to_edge(p, vertices[3], vertices[0])
    
    if is_in_v1(p, vertices):
        return -compute_distance(p, vertices[0])
    
    if is_in_v2(p, vertices):
        return -compute_distance(p, vertices[1])
    
    if is_in_v3(p, vertices):
        return -compute_distance(p, vertices[2])
    
    if is_in_v4(p, vertices):
        return -compute_distance(p, vertices[3])

def objective_function_with_constraints(x, goalPosition=(20,20), vertices_list=None):
    '''Calculate the difference between the goal position and the current
    position of the distance'''
    res = 0
    res = np.linalg.norm(np.array([x[-2], x[-1]]) - np.array(goalPosition))**2
    # res = (x[-2] - goalPosition[0])**2 + (x[-1] - goalPosition[1])**2

    # dis = distance_function(np.array([x[-2], x[-1]]), vertices_list)
    # res += np.exp(distance_function(np.array([x[-2], x[-1]]), vertices_list)) - 1

    for i in range(0, len(x), 2):
        for vertices in vertices_list[i//2]:
            res += np.exp(distance_function(np.array([x[i], x[i+1]]), vertices)) - 1

    return res

objective_grad = jax.grad(objective_function_with_constraints)

def objective_function_with_constraints_gradient(x, goalPosition=(20,20), vertices_list=None):
    return numpy.asarray(objective_grad(x, goalPosition, vertices_list))

def testing_jax():
    x = np.array([-0.5, .25, 3, 4])
    vertices = np.array([[[[1,1],[1,0],[0,0],[0,1]], [[3,3],[3,2],[2,2],[2,3]]],
                         [[[5,5],[5,4],[4,4],[4,5]], [[7,7],[7,6],[6,6],[6,7]]]])
    print(vertices.shape)
    print(objective_function_with_constraints(x, goalPosition=(10,10), vertices_list=vertices))
    print(objective_grad(x, goalPosition=(10,10), vertices_list=vertices))


if __name__ == '__main__':
    testing_jax()
    print('Done')