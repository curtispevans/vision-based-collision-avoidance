import numpy as np




class WedgeEstimator:
    def __init__(self) -> None:
        self.ts = ts_simulation
        tau = self.ts/3
        

        # this is the smallest and largest area that the intruder can be in
        self.smallest_intruder_area = min_area
        self.largest_intruder_area = max_area

        self.bearing_uncertainty = bearing_uncertainty


    def set_velocity_position(self, bearing_angles, sizes, thetas, ownship_positions):
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

        for bearing, size, theta, ownship_pos in zip(bearing_angles, sizes, thetas, ownship_positions):
            # get the range of the intruder
            intruder_min_range = self.smallest_intruder_area / size
            intruder_max_range = self.largest_intruder_area / size
            # get the line of sight
            los_body = np.array([[np.cos(bearing)], [np.sin(bearing)]])
            
            # get the positions of the intruder
            close_pos = los_body * intruder_min_range
            far_pos = los_body * intruder_max_range
            # append the positions to the lists
            close_positions.append(close_pos)
            far_positions.append(far_pos)
        
        # save the ownship last position and velocity
        self.init_own_pos = ownship_positions[-1]
        self.init_own_vel = (ownship_positions[-1] - ownship_positions[0]) / ((len(ownship_positions) - 1)*self.ts)
        
        # save the last positions and average velocities of the close position
        self.close_pos = close_positions[-1]
        close_diff = np.diff(np.array(close_positions), axis=0)[:] / self.ts
        self.close_vel = np.mean(close_diff, axis=0)

        # save the last positions and average velocities of the far position
        self.far_pos = far_positions[-1]
        far_diff = np.diff(np.array(far_positions), axis=0)[:] / self.ts
        self.far_vel = np.mean(far_diff, axis=0)

    def get_wedge_vertices(self, t):
        """
        Return the vertices of the wedge at a given time
        """
        # get the vertices of the wedge
        vertices, intruder_dir, r = get_wedge_vertices(t, self.close_pos, self.close_vel, self.far_pos, self.far_vel, self.init_own_pos, self.init_own_vel, self.bearing_uncertainty)
        ownship_pos = self.init_own_pos + self.init_own_vel * t
        vertices += np.array([[ownship_pos[1], ownship_pos[0]]])
        return vertices
    
    

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
    # min_wingspan_dist = np.linalg.norm(own_fut_pos - close_fut_pos)
    min_wingspan_dist = np.linalg.norm(close_fut_pos)
    close_lateral_dist = min_wingspan_dist * perp_factor
    close_right = close_fut_pos + close_lateral_dist
    close_left = close_fut_pos - close_lateral_dist

    # find the corners of the far intruder
    # max_wingspan_dist = np.linalg.norm(own_fut_pos - far_fut_pos)
    max_wingspan_dist = np.linalg.norm(far_fut_pos)
    far_lateral_dist = max_wingspan_dist * perp_factor
    far_right = far_fut_pos + far_lateral_dist
    far_left = far_fut_pos - far_lateral_dist

    return np.array([far_right, close_right, close_left, far_left]), intruder_dir, np.linalg.norm(intruder_dir)
    


