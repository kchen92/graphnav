import cv2
import numpy as np
import os
import yaml

from semnav.lib.utils import read_area_data


class Image2MapTransformer(object):
    """Class for transforming image pixel coordinates (row, col) to /map coordinates (x, y).
    """

    def __init__(self, yaml_filepath):
        # Load yaml file containing map data
        with open(yaml_filepath, 'r') as f:
            map_yaml = yaml.load(f)
        img_path = os.path.join(os.path.dirname(yaml_filepath), map_yaml['image'])

        self.origin = map_yaml['origin']  # World coordinates of bottom-left pixel
        self.resolution = map_yaml['resolution']  # (n_row, n_col)
        self.image_size = cv2.imread(img_path).shape[:2]  # (n_rows, n_cols)
        # Bottom left pixel in (col, row) format
        self.bottom_left_modified = np.array([0, self.image_size[0] - 1])

        # For self.get_room_name()
        mat_filepath = os.path.join(os.path.dirname(yaml_filepath), map_yaml['mat_filepath'])
        self.area_data = read_area_data(mat_filepath)
        self.id2name = {room['id']: room['name'] for room in self.area_data['rooms']}

    def pixel2map(self, px_coord):
        """Convert pixel coordinates (row, col) to /map coordinates in ROS.

        Args:
            px_coord: Nx2 array of pixel coordinates (row, col).

        Returns:
            map_coord: Nx2 array of map coordinates (x, y).
        """
        one_d = False
        if px_coord.shape == (2,):
            px_coord = px_coord.reshape((1, 2))
            one_d = True

        # Check bounds
        mins = np.amin(px_coord, axis=0)
        maxes = np.amax(px_coord, axis=0)
        assert np.all(mins >= 0)
        assert np.all(maxes < self.image_size)

        # Compute distance from bottom-left pixel
        px_coord_rolled = np.roll(px_coord, 1, axis=1)
        px_coord_rolled[:, 0] = -px_coord_rolled[:, 0]

        map_coord = (self.bottom_left_modified - px_coord_rolled).astype(np.float32) * self.resolution + self.origin[:2]
        if one_d is True:
            map_coord = map_coord.reshape((2,))
        return map_coord

    def map2pixel(self, map_coord):
        """Convert /map coordinates (x, y) to pixel coordinates.

        Args:
            map_coord: Nx2 array of map coordinates (x, y).

        Returns:
            px_coord: Nx2 array of pixel coordinates (row, col).
        """
        one_d = False
        if map_coord.shape == (2,):
            map_coord = map_coord.reshape((1, 2))
            one_d = True

        # Set origin to bottom left pixel
        centered_map_coord = map_coord - self.origin[:2]
        px_from_bottom_left = centered_map_coord.astype(np.float32) / self.resolution
        row_coord = self.image_size[0] - px_from_bottom_left[:, 1] - 1
        col_coord = px_from_bottom_left[:, 0]
        row_coord = row_coord[:, np.newaxis]
        col_coord = col_coord[:, np.newaxis]
        px_from_top_left = np.hstack((row_coord, col_coord))
        px_coord = np.rint(px_from_top_left).astype(np.int32)

        # Check bounds
        mins = np.amin(px_coord, axis=0)
        maxes = np.amax(px_coord, axis=0)
        assert np.all(mins >= 0)
        assert np.all(maxes < self.image_size)

        if one_d is True:
            px_coord = px_coord.reshape((2,))
        return px_coord

    def get_room_name(self, position, xy=False):
        """Get the room name for the given position.

        Args:
            position: xyz coordinates representing the agent position. If xy is True, then position
                is xy coordinates.
        """
        if xy is False:
            position = position[:2]  # Convert from xyz to xy
        px_coord = self.map2pixel(np.array(position))
        room_name = self.id2name.get(self.area_data['semantic_map'][px_coord[0], px_coord[1]])
        return room_name
