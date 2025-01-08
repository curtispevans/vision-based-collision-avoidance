import numpy as np
from jax import jacfwd, jacrev, jacobian
import jax.numpy as jnp
import matplotlib.pyplot as plt

intruder_wingspan = 10

def motion_model(x, delta_t):
    '''
    x: state vector
    delta_t: time step
    '''
    bearing_dot_relative_velocity = x[0]*x[3] + x[1]*x[4]
    f = jnp.array([x[1]**2*x[3] - x[0]*x[1]*x[4],
                  -x[0]*x[1]*x[3] + x[0]**2*x[4],
                  -2*x[2]*bearing_dot_relative_velocity,
                   -x[3]*bearing_dot_relative_velocity,
                   -x[4]*bearing_dot_relative_velocity,
                   0,
                   0])
    noise = jnp.array([0, 0, 0, np.random.normal(0, 0.1), np.random.normal(0, 0.1), 0, 0])
    return x + (f + noise)*delta_t

def jacobian_jax(x, delta_t):
    return jacfwd(motion_model)(x, delta_t)

def measurement_model(x):
    H = jnp.array([[1,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0],
                   [0,0,1,0,0,0,0]])
    return jnp.array([x[0], x[1], x[2]]), H

def kalman_update(mu, sigma, measurement, R, Q, delta_t):
    # Prediction
    mu_bar = motion_model(mu, delta_t)
    J = jacobian_jax(mu, delta_t)
    sigma_bar = J@sigma@J.T + R

    # Update
    z, H = measurement_model(mu_bar)
    S = H@sigma_bar@H.T + Q
    K = sigma_bar@H.T@np.linalg.inv(S)
    mu = mu_bar + K@(measurement - z)
    sigma = (jnp.eye(len(K)) - K@H)@sigma_bar

    return mu, sigma 
    
def get_ownship_and_intruders_from_filepath(filepath):
    real = np.load(filepath)
    ownship = real[0]
    intruders = []
    for intruder in real[1:]:
        intruders.append(intruder)

    start = ownship[0]
    middle = start - np.array([100., 0.])
    maneuver = [start]
    
    for i in range(0, 299, 1):
        theta = np.pi * i/180
        point = middle + 100*np.array([np.cos(theta), np.sin(theta)])
        maneuver.append(point)

    maneuver = np.array(maneuver)
    print(len(maneuver), len(ownship))
    plt.plot(ownship[:,0], ownship[:,1], 'o')
    plt.plot(maneuver[:,0], maneuver[:,1], 'o')
    plt.axis('equal')
    plt.show()
    return ownship, maneuver, intruders

def calculate_bearing_pixel_size(ownship, intruders):
    bearings = []
    pixel_sizes = []
    num_intruders = len(intruders)
    for i in range(num_intruders):
        bearings.append([])
        pixel_sizes.append([])
        for own_pos, intruder_pos in zip(ownship, intruders[i]):
            true_bearing = np.arctan2(intruder_pos[0] - own_pos[0], intruder_pos[1] - own_pos[1])
            rho = np.linalg.norm(intruder_pos - own_pos)
            size = intruder_wingspan / rho
            bearings[i].append(true_bearing)
            pixel_sizes[i].append(size)

    return bearings, pixel_sizes



def testing_jacobian():
    x = jnp.array([jnp.cos(1),jnp.sin(1),1,1,1,1,1])
    delta_t = 1
    print(jacobian_jax(x, delta_t))


def testing_kalman_update():
    intruder = jnp.load('visualAvoidance2D/data/intruder2.npy')
    mu = jnp.array([jnp.cos(intruder[0,0]), jnp.sin(intruder[0,0]), intruder[0,1], 10, -30, 75, 10])
    sigma = jnp.eye(7)*0.1

    x_pos = []
    y_pos = []
    sizes = []
    angles = []

    R = jnp.eye(7)*0.1
    Q = jnp.eye(3)*0.1
    delta_t = 1/25

    for row in intruder:
        measurement = jnp.array([jnp.cos(row[0]), jnp.sin(row[0]), row[1]])        
        mu, sigma = kalman_update(mu, sigma, measurement, R, Q, delta_t)
        x_pos.append(mu[0])
        y_pos.append(mu[1])
        sizes.append(mu[2])
        angles.append(np.arctan2(mu[1], mu[0]))
        # print(mu[:2].T @ measurement[:2])
        # print(mu[-2:])

    print(mu)

    plt.polar(angles, np.ones_like(angles), 'o', label='Estimated')
    plt.polar(intruder[:,0], np.ones_like(intruder[:,0]), 'o', label='Measured')
    plt.legend()
    plt.show()

    plt.plot(sizes, label='Estimated')
    plt.plot(intruder[:,1], label='Measured')
    plt.legend()
    plt.show()
    
        

if __name__ == "__main__":
    filepath = "visualAvoidance2D/data/xplane_data/0002/20241205_151830_all_positions_in_path.npy"
    ownship, maneuver, intruders = get_ownship_and_intruders_from_filepath(filepath)
    bearings, pixel_sizes = calculate_bearing_pixel_size(maneuver, intruders)
    
    bearing1 = bearings[0]
    size1 = pixel_sizes[0]
    mu = jnp.array([jnp.cos(bearing1[0]), jnp.sin(bearing1[0]), size1[0], 0, 0, 75, 2])
    sigma = jnp.eye(7)*0.1

    R = jnp.eye(7)*0.1
    Q = jnp.eye(3)*0.1
    delta_t = 1/30

    for i in range(1, len(bearing1)):
        bearing = bearing1[i]
        size = size1[i]
        measurement = jnp.array([jnp.cos(bearing), jnp.sin(bearing), size])
        mu, sigma = kalman_update(mu, sigma, measurement, R, Q, delta_t)
    
    print(mu[-2:])
