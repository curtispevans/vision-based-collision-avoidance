import numpy as np
from jax import jacfwd, jacrev, jacobian
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
    
        

testing_kalman_update()