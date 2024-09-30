import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt

############################################################################################################
# global variables for the simulation
ts_simulation = 1/25

uav_scale = 2
# this is not to be changed - it is the pixel wingspan of the fixed wing vehicle as-drawn
uav_wingspan = 24
uav_size = uav_scale * uav_wingspan

# parameters for the wedge estimator
bearing_uncertainty = 0.1

# This is the smallest pixel area that an intruder could possibly be
min_area = 30
# This is the largest pixel area that an intruder could possibly be
max_area = 100

############################################################################################################


class MedianFilter:
    def __init__(self, window_size):
        '''
        Parameters:
        window_size: int
        '''
        self.window_size = window_size

    def fit(self, X):
        '''
        Parameters:
        X: np.array of shape (n,)
        '''
        # print(len(X) <= self.window_size-1)
        if not len(X) <= self.window_size - 1:
            raise ValueError(f'Window size is too small array must be at least {self.window_size - 1} long')
        self.X = X
    
    def predict(self, x):
        '''
        Parameters:
        x: float
        Returns:
        y: float
        '''
        self.X.append(x)
        y = np.median(self.X)
        self.X = self.X[1:]
        return y

class MsgState:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
        
    def __init__(self,
                 pos=np.array([[0.], [0.]]),
                 vel=0.,
                 theta=0.,
                 ):
        self.pos = pos  # position in inertial frame
        self.vel = vel  # speed
        self.theta = theta  # angle from north
    
    def print(self):
        print('position=', self.pos,
              'velocity=', self.vel,
              'theta=', self.theta)


class MavDynamics:
    def __init__(self, x0):
        self._ts_simulation = ts_simulation
        self._state = np.array([[x0.pos[0, 0]],  # (0)
                               [x0.pos[1, 0]],   # (1)
                               [x0.theta],   # (2)
                               [x0.vel],  # (3)
                                ])
        # initialize true_state message
        self.state = MsgState(pos=x0.pos,
                              vel=x0.vel,
                              theta=x0.theta)

    ###################################
    # public functions
    def update(self, u):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state[0:4], u)
        k2 = self._derivatives(self._state[0:4] + time_step / 2. * k1, u)
        k3 = self._derivatives(self._state[0:4] + time_step / 2. * k2, u)
        k4 = self._derivatives(self._state[0:4] + time_step * k3, u)
        self._state[0:4] += time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # update the message class for the true state
        self._update_true_state()
    
    # This function is only used for testing purposes
    def camera(self, intruders):
        bearings = []  # angles to intruders (rad)
        sizes = []  # sizes of intruder on camera (rad)
        for intruder in intruders:
            los = intruder.state.pos - self._state[0:2]
            # get the bearing angle from the ownship to the intruder
            bearing = np.arctan2(los[1, 0], los[0, 0]) - self._state[2, 0]

            # get the pixel size of the intruder
            size = uav_size / (np.linalg.norm(los) + .01)

            bearings.append(bearing)
            sizes.append(size)
        return bearings, sizes

    ###################################
    # private functions
    def _derivatives(self, state, u):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        north_dot = state[3, 0] * np.cos(state[2, 0])
        east_dot = state[3, 0] * np.sin(state[2, 0])
        theta_dot = u
        vel_dot = 0.
        # collect the derivative of the states
        x_dot = np.array([[north_dot, east_dot, theta_dot, vel_dot]]).T
        return x_dot

    def _update_true_state(self):
        self.state.pos = self._state[0:2]
        self.state.theta = self._state[2, 0]
        self.state.vel = self._state[3, 0]


class GMM:
    def __init__(self, n_components, weights=None, means=None, covars=None):
        """
        Initializes a GMM.
        
        The parameters weights, means, and covars are optional. If fit() is called,
        they will be automatically initialized from the data.
        
        If specified, the parameters should have the following shapes, where d is
        the dimension of the GMM:
            weights: (n_components,)
            means: (n_components, d)
            covars: (n_components, d, d)
        """
        self.n_components = n_components
        self.weights = weights
        self.means = means
        self.covars = covars
    

    def component_logpdf(self, k, z):
        """
        Returns the logarithm of the component pdf. This is used in several computations
        in other functions.
        
        Parameters:
            k (int) - the index of the component
            z ((d,) or (..., d) ndarray) - the point or points at which to compute the pdf
        Returns:
            (float or ndarray) - the value of the log pdf of the component at 
        """
        return np.log(self.weights[k]) + st.multivariate_normal.logpdf(z, self.means[k], self.covars[k])
        
    def pdf(self, z):
        """
        Returns the probability density of the GMM at the given point or points.
        
        Parameters:
            z ((d,) or (..., d) ndarray) - the point or points at which to compute the pdf
        Returns:
            (float or ndarray) - the value of the GMM pdf at z
        """
        return sum([self.weights[k]*st.multivariate_normal.pdf(z, self.means[k], self.covars[k]) for k in range(self.n_components)])
    
    def cdf(self, z):
        """
        Returns the probability of [a1,b1]x...x[an,bn]
        Parameters:
            z((n,n) ndarray) the bounds to compute the probability
        Returns:
            (float or ndarray) - the probability of the GMM in those bounds """
        return sum([self.weights[k]*st.multivariate_normal.cdf(z, self.means[k], self.covars[k]) for k in range(self.n_components)])
    


class WedgeEstimator:
    def __init__(self, n=15) -> None:
        self.ts = ts_simulation

        # this is the smallest and largest area that the intruder can be in
        self.smallest_intruder_area = min_area
        self.largest_intruder_area = max_area

        self.bearing_uncertainty = bearing_uncertainty

        self.n = n
        self.cov = np.array([[6,0],[0,1]])
        


    def set_velocity_position(self, bearing_angles, sizes, thetas, ownship_positions, ownship_state):
        """Sets the position and velocity of the intruder based on two measurements
        
        Parameters:
            bearing_angles (list) - the bearing angles of the intruders
            sizes (list) - the sizes of the intruders
            thetas (list) - the headings of the ownship
            ownship_positions (list) - the positions of the ownship
        """
        # make empty lists for the positions
        close_positions = []
        far_positions = []

        # if len(bearing_angles) > 10:
        #     bearing_angles = bearing_angles[-10:]
        #     sizes = sizes[-10:]
        #     thetas = thetas[-10:]
        #     ownship_positions = ownship_positions[-10:]


        for bearing, size, theta, ownship_pos in zip(bearing_angles, sizes, thetas, ownship_positions):
            # get the range of the intruder
            intruder_min_range = self.smallest_intruder_area / size
            intruder_max_range = self.largest_intruder_area / size
            # get the line of sight
            los_body = np.array([[np.cos(bearing)], [np.sin(bearing)]])
            # rotate the line of sight to the inertial frame
            R = rotation_matrix(theta)
            los_interial = R @ los_body
            # get the positions of the intruder
            close_pos = ownship_pos + los_interial * intruder_min_range
            far_pos = ownship_pos + los_interial * intruder_max_range
            # append the positions to the lists
            close_positions.append(close_pos)
            far_positions.append(far_pos)
        
        # save the ownship last position and velocity
        self.init_own_pos = ownship_positions[-1]
        heading = np.array([[np.cos(ownship_state.theta)], [np.sin(ownship_state.theta)]])
        self.init_own_vel = heading * ownship_state.vel

        # save the last positions and average velocities of the close position
        self.close_pos = close_positions[-1]
        close_diff = np.diff(np.array(close_positions), axis=0) / self.ts
        self.close_vel = np.mean(close_diff, axis=0)

        # save the last positions and average velocities of the far position
        self.far_pos = far_positions[-1]
        far_diff = np.diff(np.array(far_positions), axis=0) / self.ts
        self.far_vel = np.mean(far_diff, axis=0)


    def get_wedge_vertices(self, t):
        """
        Return the vertices of the wedge at a given time
        """
        # get the vertices of the wedge
        vertices, intruder_dir, r = get_wedge_vertices(t, self.close_pos, self.close_vel, self.far_pos, self.far_vel, self.init_own_pos, self.init_own_vel, self.bearing_uncertainty)
        
        return vertices
    
    def get_wedge_single_gaussian(self, t):
        '''
        Return the elongated gaussian for the wedge at a given time
        '''
        # get the vertices of the wedge
        vertices, intruder_dir, r = get_wedge_vertices(t, self.close_pos, self.close_vel, self.far_pos, self.far_vel, self.init_own_pos, self.init_own_vel, self.bearing_uncertainty)
        middle_bottom = (vertices[1] + vertices[2]) / 2
        middle_left = (vertices[2] + vertices[3]) / 2
        middle_right = (vertices[0] + vertices[1]) / 2
        perp_direction = middle_right - middle_left
        perp_dist = np.linalg.norm(perp_direction)
        middle = middle_bottom + 0.5*intruder_dir

        dot_prod = (intruder_dir.T @ np.array([[1.], [0.]]))[0][0]
        oriented_bearing = np.arccos(dot_prod/np.linalg.norm(intruder_dir))
        cross_prod = cross_product(intruder_dir, np.array([[1.], [0.]]))
        if cross_prod > 0:
            oriented_bearing = -oriented_bearing
        
        # get the rotation matrix
        R = rotation_matrix(oriented_bearing)

        cov = R @ np.diag([(r/2)**2,(perp_dist/2)**2]) @ R.T
        gaussian = st.multivariate_normal(mean=middle.flatten(), cov=cov)

        return gaussian


    def get_wedge_gmm(self, t):
        """
        Returns the wedge at the given time.
        
        Parameters:
            t (float) - the time of the wedge
        Returns:
            gmm (GMM) - the GMM representing the wedge at the given time
        """
        # get the vertices of the wedge
        vertices, intruder_dir, r = get_wedge_vertices(t, self.close_pos, self.close_vel, self.far_pos, self.far_vel, self.init_own_pos, self.init_own_vel, self.bearing_uncertainty)
        

        dot_prod = (intruder_dir.T @ np.array([[1.], [0.]]))[0][0]
        oriented_bearing = np.arccos(dot_prod/np.linalg.norm(intruder_dir))
        cross_prod = cross_product(intruder_dir, np.array([[1.], [0.]]))
        if cross_prod > 0:
            oriented_bearing = -oriented_bearing
        
        # get the rotation matrix
        R = rotation_matrix(oriented_bearing)
        # rotate the vertices for better uniform points
        oriented_vertices = (R.T @ vertices.T).T
        
        # get the points
        n = self.n
        points = faster_equally_spaced_points(*oriented_vertices, n)
        # rotate the points back 
        points = (R @ points.T).T
        n_points = len(points)
        
        # get the covariance and weights then generate the GMM
        cov = R @ self.cov @ R.T
        weights = np.ones(n_points) / n_points
        gmm = GMM(n_components=n_points, weights=weights, means=points, covars=[7*r*cov]*n_points)

        return gmm




############################################################################################################
# Helper and public functions
############################################################################################################


def rotation_matrix(theta):
    """
    Returns the 2D rotation matrix for the given angle.
    
    Parameters:
        theta (float) - the angle of rotation in radians
    Returns:
        (2, 2) ndarray - the rotation matrix
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def get_wedge_vertices(t, close_pos, close_vel, far_pos, far_vel, init_own_pos, init_own_vel, bearing_uncertainty):
    """
    Returns the wedge at the given time.
    
    Parameters:
        t (float) - the time of the wedge
        close_pos (2,) ndarray - the position of the close intruder
        close_vel (2,) ndarray - the velocity of the close intruder
        far_pos (2,) ndarray - the position of the far intruder
        far_vel (2,) ndarray - the velocity of the far intruder
        init_own_pos (2,) ndarray - the initial position of the ownship
        init_own_vel (2,) ndarray - the initial velocity of the ownship
        bearing_uncertainty (float) - the uncertainty in the bearing angle
    
    Returns:
        (4,2) ndarray - the vertices of the wedge [far_right, close_right, close_left, far_left]
    """

    # get the future positions of the intruders
    close_fut_pos = close_pos + t * close_vel
    far_fut_pos = far_pos + t * far_vel

    # get the future position of the ownship
    own_fut_pos = init_own_pos + t * init_own_vel

    # get the direction of the intruder
    intruder_dir = far_fut_pos - close_fut_pos
    intruder_dir_unit = intruder_dir / np.linalg.norm(intruder_dir)
    intruder_dir_perp = np.array([intruder_dir_unit[1], -intruder_dir_unit[0]])

    perp_factor = intruder_dir_perp * np.tan(bearing_uncertainty)

    # find the corners of the close intruder
    min_wingspan_dist = np.linalg.norm(own_fut_pos - close_fut_pos)
    close_lateral_dist = min_wingspan_dist * perp_factor
    close_right = close_fut_pos + close_lateral_dist
    close_left = close_fut_pos - close_lateral_dist

    # find the corners of the far intruder
    max_wingspan_dist = np.linalg.norm(own_fut_pos - far_fut_pos)
    far_lateral_dist = max_wingspan_dist * perp_factor
    far_right = far_fut_pos + far_lateral_dist
    far_left = far_fut_pos - far_lateral_dist

    return np.array([far_right, close_right, close_left, far_left]), intruder_dir, np.linalg.norm(intruder_dir)



def faster_equally_spaced_points(v1, v2, v3, v4, n):
    """
    Generates n equally spaced points inside the given vertices.
    
    Parameters:
        v1, v2, v3, v4 : plot_wedge(vertices_list[i], ax)
            Number of points to generate inside the vertices.
    
    Returns:
        numpy.ndarray
            Array of shape (n, 2) containing the generated points.
    """

    # Function to check if points are inside the trapezoid using the cross product method
    def are_inside_wedge(points):
        # for this function to work a and b must be (2,) and o must be (n,2)
        def sign(a, b, o):
            # get the determinate of [a-o, b-o]
            return (a[0] - o[:,0]) * (b[1] - o[:,1]) - (b[0] - o[:,0]) * (a[1] - o[:,1])
        
        d1 = sign(v1, v2, points)
        d2 = sign(v2, v3, points)
        d3 = sign(v3, v4, points)
        d4 = sign(v4, v1, points)
        
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0) | (d4 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0) | (d4 > 0)
        
        return ~(has_neg & has_pos)
    

    # Generate n evenly spaced points along each axis
    north_range = np.linspace(min(v1[0], v2[0], v3[0], v4[0]), max(v1[0], v2[0], v3[0], v4[0]), n)
    east_range = np.linspace(min(v1[1], v2[1], v3[1], v4[1]), max(v1[1], v2[1], v3[1], v4[1]), n)
    
    # Generate the grid of points
    N, E = np.meshgrid(north_range, east_range)
    points = np.column_stack((N.ravel(), E.ravel()))
    
    # Filter the points to keep only those inside the trapezoid
    inside = are_inside_wedge(points)
    points = points[inside]
    
    return points

def plot_wedge(vertices, ax):
    ax.plot((vertices[0,1],vertices[1,1]),(vertices[0,0],vertices[1,0]),'r-')
    ax.plot((vertices[1,1],vertices[2,1]),(vertices[1,0],vertices[2,0]),'r-')
    ax.plot((vertices[2,1],vertices[3,1]),(vertices[2,0],vertices[3,0]),'r-')
    ax.plot((vertices[3,1],vertices[0,1]),(vertices[3,0],vertices[0,0]),'r-')

def cross_product(a, b):
    return a.item(0)*b.item(1) - a.item(1)*b.item(0)

