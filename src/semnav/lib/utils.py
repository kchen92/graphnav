"""Utility functions.
"""

from __future__ import print_function
from __future__ import division

import cv2
import glob
import numpy as np
import os
import scipy.io as sio


def read_area_data(mat_filepath):
    """Read the mat data.

    Area data:
        resolution: Float.
        clutter_map: Numpy array with map size and type np.uint8 with values in {0, 1}.
        occupancy_map: Numpy array with map size and type np.uint8 in range [0, 255].
        semantic_map: Numpy array with map size and type np.uint8.
        rooms: List of dicts.

    Args:
        mat_filepath: Path to mat file.

    Returns:
        area_data: Dict containing area data.
    """
    mat_data = sio.loadmat(mat_filepath)
    mat_area_data = mat_data['area_data'][0, 0]
    mat_rooms = mat_area_data['rooms']
    rooms = []
    for room_idx in range(mat_rooms.shape[1]):
        cur_room = mat_rooms[0, room_idx]
        cur_bb = {
            'upper_left_px': cur_room['bounding_box'][0, 0]['upper_left_px'][0].astype(np.int32),
            'lower_right_px': cur_room['bounding_box'][0, 0]['lower_right_px'][0].astype(np.int32),
        }
        cur_room_dict = {
            'name': cur_room['name'][0].encode('ascii', 'ignore'),
            'id': int(cur_room['id'][0][0]),
            'points_px': cur_room['points'].astype(np.int32),
            'bounding_box': cur_bb,
        }
        rooms.append(cur_room_dict)

    area_data = {
        'occupancy_map': mat_area_data['occupancy_grid'],
        'clutter_map': mat_area_data['clutter_grid'],
        'semantic_map': mat_area_data['semantic_map'],
        'resolution': float(mat_area_data['resolution'][0, 0]),
        'rooms': rooms,
        }
    assert area_data['occupancy_map'].shape == area_data['clutter_map'].shape
    assert area_data['occupancy_map'].shape == area_data['semantic_map'].shape

    return area_data


def dilate_map(cur_map, kernel_shape, iterations=1):
    """Perform a dilation on the image/map.

    Args:
        cur_map: 2D map of shape (height, width) with dtype uint8.
        kernel_shape: Shape of rectangular kernel (e.g. [5, 5]).

    Returns:
        dilated_map: Dilated map/image.
    """
    kernel = np.ones(kernel_shape, np.uint8)
    dilation = cv2.dilate(cur_map, kernel, iterations=iterations)
    return dilation


def compute_orientation(start_px_coord, end_px_coord, noise=0):
    """Compute the general direction/orientation from the start point to the end point. Assumes
    coordinate frame is 'map'. If noise > 0, noise will be added to the computed orientation.

    The general direction/orientation can be one of the following:
        - Right in image map (0 radians): Towards positive X-axis
        - Upwards in image map (1.57 radians): Towards positive Y-axis
        - Left in image map (3.14 radians): Away from positive X
        - Downwards in image map (4.71 radians): Negative Y-axis direction

    Args:
        start_px_coord: Start point in pixel coordinates as ndarray.
        end_px_coord: End point in pixel coordinates as ndarray.
        noise: The plus/minus delta that is added to the final computed orientation in DEGREES.

    Returns:
        theta: Euler angle theta wrt z-axis.
    """
    if noise < 0:
        raise ValueError('Noise must be nonnegative.')

    diff = end_px_coord - start_px_coord
    abs_diff = np.abs(diff)
    if abs_diff[0] > abs_diff[1]:  # Either upwards or downwards
        if diff[0] < 0:  # Up
            theta = np.pi / 2
        else:  # Down
            theta = 3 * np.pi / 2
    else:  # Either left or right
        if diff[1] < 0:  # Left
            theta = np.pi
        else:  # Right
            theta = 0.

    if noise > 0:
        # Convert noise to radians
        noise_radians = float(noise) / 180 * np.pi

        # Sample
        delta = np.random.uniform(low=-noise_radians, high=noise_radians)

        # Add to theta
        theta += delta

    return theta


def mkdir(cur_dir, verbose=False):
    """Make the directory if it doesn't exist.
    """
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)

        if verbose:
            print('Making directory:', cur_dir)


def compute_angle_delta(initial_angle, final_angle):
    delta = final_angle - initial_angle
    if np.abs(delta) > np.pi:
        if final_angle < 0.:
            delta = delta + 2. * np.pi
        else:
            delta = delta - 2. * np.pi
    return delta


def compute_dist(map_coord1, map_coord2):
    """Given two map coordinates, compute the Euclidean distance.
    """
    return np.linalg.norm(map_coord2 - map_coord1)


def compute_dist_to_node(position, node):
    """Given a position and node, compute the planar (xy) distance from position to node as
    Euclidean distance. The distance scales inversely with the node's node_size.

    Args:
        position: Position (e.g. cur_frame['gt_odom']['position']). This should be a (3,) np.ndarray.
        node: Node.
    """
    return compute_dist(position[:2], node.map_coord) / node.node_size


def sigmoid_np(x):
    return 1. / (1. + np.exp(-x))


def load_frame(frame_path):
    with np.load(frame_path) as raw_data:
        frame_data = raw_data['arr_0'].item()
    return frame_data


def get_frame_paths(sequence_dir):
    """Return a list of frame filepaths for the given sequence.

    Args:
        sequence_dir: Full path to the sequence directory.

    Returns:
        frame_paths: List of (full) filepaths to frames in the sequence in sorted order.
    """
    glob_input = os.path.join(sequence_dir, '*.npz')
    frame_fnames = sorted(glob.glob(glob_input))
    frame_paths = [os.path.join(sequence_dir, f) for f in frame_fnames]
    return frame_paths


def get_frame_idx(frame_path):
    """Get the index of the frame (1-based indexing).

    Args:
        frame_path: Path to the frame (.npz). This can be the full path or just the filename.
    """
    return int(os.path.splitext(frame_path)[0].split('_')[-1])
