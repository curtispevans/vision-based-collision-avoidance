import numpy as np
import matplotlib.pyplot as plt

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
    if (right < 0 and bottom > 0 and left < 0 and top < 0) or (right > 0 and bottom > 0 and left > 0 and top < 0):
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
    

def compute_cross_product2D_vectorized(a, b):
    return a[:,0]*b[:,1] - a[:,1]*b[:,0]

def compute_relevant_cross_products_vectorized(points, vertices):
    '''The vertices come in a list of 4 points, with the first point being the top right point of the wedge, 
       the second point being the bottom right point of the wedge, the third point being the bottom left point of the wedge,
       and the fourth point being the top left point of the wedge.'''
    right = compute_cross_product2D_vectorized(vertices[0]-points, vertices[1]-points)
    bottom = compute_cross_product2D_vectorized(vertices[1]-points, vertices[2]-points)
    left = compute_cross_product2D_vectorized(vertices[2]-points, vertices[3]-points)
    top = compute_cross_product2D_vectorized(vertices[3]-points, vertices[0]-points)
    return right, bottom, left, top

def is_inside_wedge_vectorized(points, vertices):
    '''The vertices come in a list of 4 points, with the first point being the top right point of the wedge, 
       the second point being the bottom right point of the wedge, the third point being the bottom left point of the wedge,
       and the fourth point being the top left point of the wedge.'''
    right, bottom, left, top = compute_relevant_cross_products_vectorized(points, vertices)
    inside = np.logical_and(np.logical_and(right < 0, bottom < 0), np.logical_and(left < 0, top < 0))
    return inside

def distance_constraint(x, vertice_list):
    distances = np.zeros(len(x)//2)

    for i in range(0, len(x), 2):
        for vertices in vertice_list[i//2]:
            distances[i//2] += distance_function(np.array([x[i], x[i+1]]), vertices)

    return distances

def distance_constraint_vectorized(x, vertice_list):
    num_intruders = len(vertice_list[0])
    distances = np.zeros(num_intruders*len(x)//2)

    for i in range(0, len(x), 2):
        for j, vertices in enumerate(vertice_list[i//2]):
            distances[num_intruders*i//2 + j] = distance_function(np.array([x[i], x[i+1]]), vertices)

    # print(distances)
    return distances

def probability_constraint(x, wedges):
    sim_time = 0
    pdf_values = []
    for i in range(0, len(x), 2):
        value = 0
        for wedge in wedges:
            value += wedge.get_wedge_gmm(i//2).pdf(np.array([x[i], x[i+1]]))
        pdf_values.append(value)
    
    return pdf_values