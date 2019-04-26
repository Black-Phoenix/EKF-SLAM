from __future__ import division
import numpy as np
import slam_utils
from scipy.stats.distributions import chi2
import tree_extraction


def motion_model(u, dt, ekf_state, vehicle_params):
    '''
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.

        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    '''
    # motion = np.array([ekf_state['x'][0] + dt * u[0] * np.cos(ekf_state['x'][2]),
    #                    ekf_state['x'][1] + dt * u[0] * np.sin(ekf_state['x'][2]),
    #                    ekf_state['x'][2] + dt * u[0] * np.tan(u[1])]).T
    # G = np.array([[1, 0, -dt*u[0]*np.sin(ekf_state['x'][2])],
    #      [0, 1, dt*u[0]*np.cos(ekf_state['x'][2])],
    #      [0, 0, 1]])
    x, y, p = ekf_state['x'][:3]
    L = vehicle_params['L']
    a = vehicle_params['a']
    H = vehicle_params['H']
    b = vehicle_params['b']
    v, alpha = u
    v = v / (1 - np.tan(alpha) * H / L)
    motion = np.array([x + dt * (v * np.cos(p) - v / L * np.tan(alpha) * (a * np.sin(p) + b * np.cos(p))),
                       y + dt * (v * np.sin(p) + v / L * np.tan(alpha) * (a * np.cos(p) - b * np.sin(p))),
                       p + dt * v / L * np.tan(alpha)]).T
    G = np.array([[1, 0, -dt * (v * np.sin(p) + (v * np.tan(alpha) * (a * np.cos(p) - b * np.sin(p))) / L)],
                  [0, 1, dt * (v * np.cos(p) - (v * np.tan(alpha) * (b * np.cos(p) + a * np.sin(p))) / L)],
                  [0, 0, 1]])
    return motion, G


def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    '''
    Perform the propagation step of the EKF filter given an odometry measurement u
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''

    motion, G = motion_model(u, dt, ekf_state, vehicle_params)
    ekf_state['x'][:3] = motion
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
    ekf_state['P'][:3, :3] = np.matmul(np.matmul(G, ekf_state['P'][:3, :3]), G.T) + \
                             np.diag([sigmas['xy'] ** 2, sigmas['xy'] ** 2, sigmas['phi'] ** 2])  # + \
    # 10**-6*np.eye(3)
    ekf_state['P'] = fix_cov(ekf_state['P'])
    return ekf_state


def fix_cov(P):
    offset = 10 ** -6
    P = P + (offset) * np.eye(P.shape[0])
    P = 1 / 2 * (P + P.T)
    return P


def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''
    h = ekf_state['x'][:2]
    H = np.array([[1, 0, 0], [0, 1, 0]])
    Q = np.diag([sigmas['gps'] ** 2, sigmas['gps'] ** 2])
    S_inv = np.linalg.inv(np.matmul(np.matmul(H, ekf_state['P'][:3, :3]), H.T) + Q)
    r = gps - h
    if np.abs(np.matmul(np.matmul(r.T, S_inv), r)) > 13.8:
        return ekf_state
    K = np.matmul(np.matmul(ekf_state['P'][:3, :3], H.T), S_inv)
    ekf_state['x'][:3] = ekf_state['x'][:3] + np.matmul(K, r)
    ekf_state['P'][:3, :3] = np.matmul(np.eye(3) - np.matmul(K, H), ekf_state['P'][:3, :3])  # + 10**-6*np.eye(3)
    ekf_state['P'][:3, :3] = fix_cov(ekf_state['P'][:3, :3])
    return ekf_state


def laser_measurement_model(ekf_state, landmark_id):
    '''
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian.

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''

    x, y, p = ekf_state['x'][:3]
    x_t, y_t = ekf_state['x'][3 + landmark_id * 2:3 + (landmark_id + 1) * 2]
    H = np.zeros((2, ekf_state['x'].shape[0]))
    zhat = np.array(
        [np.sqrt((x_t - x) ** 2 + (y_t - y) ** 2), slam_utils.clamp_angle(np.arctan2(y_t - y, x_t - x) - p)])
    H[:, :3] = np.array([[(x - x_t) / zhat[0], (y - y_t) / zhat[0], 0],
                         [-(y - y_t) / zhat[0] ** 2, (x - x_t) / zhat[0] ** 2, -1]])
    # middle_H = np.array([[-(2 * x - 2 * x_t) / (2 * ((x - x_t) ** 2 + (y - y_t) ** 2) ** (1 / 2)),
    #                       -(2 * y - 2 * y_t) / (2 * ((x - x_t) ** 2 + (y - y_t) ** 2) ** (1 / 2))]
    #                         , [(y - y_t) / ((x - x_t) ** 2 + (y - y_t) ** 2),
    #                            -(x - x_t) / ((x - x_t) ** 2 + (y - y_t) ** 2)]])
    H[:, 3 + landmark_id * 2:3 + (landmark_id + 1) * 2] = -H[:2, :2]
    return zhat, H


def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.
    '''
    tree_pos = slam_utils.tree_to_global_xy(np.asarray(tree)[np.newaxis, :], ekf_state).flatten()

    ekf_state['x'] = np.concatenate((ekf_state['x'], tree_pos))
    ekf_state['P'] = np.pad(ekf_state['P'], (0, 2), 'constant', constant_values=(0))
    ekf_state['P'][-1, -1] = 1
    ekf_state['P'][-2, -2] = 1
    return ekf_state


def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''
    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurement
        return [-1 for m in measurements]
    R = np.array([[sigmas['range'] ** 2, 0], [0, sigmas['bearing'] ** 2]])
    m = ekf_state['num_landmarks']
    measurement_size = len(measurements)
    # To hold the cost. rows are old measurements and columns are new ones
    M = np.zeros((measurement_size, m))
    # todo vectorize this
    measurements_np = np.array(measurements)[:, :2]
    for j in range(m):
        curr_pos, H = laser_measurement_model(ekf_state, j)
        S = np.matmul(np.matmul(H, ekf_state['P']), H.T) + R
        for i in range(measurement_size):
            # x_L, y_L = slam_utils.tree_to_global_xy(measurements_np[np.newaxis, i], ekf_state)
            # x, y = slam_utils.tree_to_global_xy(curr_pos[np.newaxis,:], ekf_state)
            # M[i,j] = np.sqrt((x_L - x)**2 + (y_L - y)**2)
            r = measurements_np[i] - curr_pos
            M[i, j] = np.matmul(np.matmul(r.T, np.linalg.inv(S)), r)
    matches = slam_utils.solve_cost_matrix_heuristic(M.copy())
    assoc = [-1] * measurement_size
    for i in matches:
        if M[i] > 15:
            assoc[i[0]] = -1
        elif M[i] > chi2.ppf(.995, df=2):
            assoc[i[0]] = -2
        else:
            assoc[i[0]] = i[1]
    return assoc


def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Perform a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''
    for i, pair in enumerate(assoc):
        if pair == -1:
            initialize_landmark(ekf_state, trees[i])
            ekf_state['num_landmarks'] += 1
        elif pair == -2:
            continue
        else:
            curr_measurement = trees[i][:2]
            h, H = laser_measurement_model(ekf_state, pair)
            Q = np.diag([sigmas['range'] ** 2, sigmas['bearing'] ** 2])
            S_inv = np.linalg.inv(np.matmul(np.matmul(H, ekf_state['P']), H.T) + Q)
            r = curr_measurement - h
            if np.abs(np.matmul(np.matmul(r.T, S_inv), r)) > 13.8:
                continue
            K = np.matmul(np.matmul(ekf_state['P'], H.T), S_inv)
            ekf_state['x'] = ekf_state['x'] + np.matmul(K, r)
            ekf_state['P'] = np.matmul(np.eye(ekf_state['P'].shape[0]) - np.matmul(K, H),
                                       ekf_state['P'])  # + 10**-6*np.eye(3)
    ekf_state['P'] = fix_cov(ekf_state['P'])
    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }

    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)

        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)

        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3, :3])))
        state_history['t'].append(t)

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key=lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50,
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75,  # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": True,
        "plot_map_covariances": False

        # Add other parameters here if you need to...
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5 * np.pi / 180,
        # Measurement noise
        "gps": 3,
        "range": 0.5,
        "bearing": 5 * np.pi / 180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array([gps[0, 1], gps[0, 2], 36 * np.pi / 180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)


if __name__ == '__main__':
    main()
