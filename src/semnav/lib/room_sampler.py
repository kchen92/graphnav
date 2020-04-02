"""Sample locations within a room.
"""

# from __future__ import print_function
from __future__ import division

import numpy as np


class RoomSampler(object):
    """Sample spawn locations in the room.
    """

    START_PT_MIN_DIST_FROM_LONG_EDGE = 0.20  # Meters
    START_PT_MIN_DIST_FROM_SHORT_EDGE = 0.20  # Meters

    def __init__(self, semantic_map, clutter_map, room_dict, resolution):
        self.name = room_dict['name']
        self.id = room_dict['id']
        self.bounding_box = room_dict['bounding_box']
        self.points_px = room_dict['points_px']
        self.semantic_map = semantic_map
        self.clutter_map = clutter_map
        self.resolution = resolution

        self.n_pixel_horz = (self.bounding_box['lower_right_px'][1]
                             - self.bounding_box['upper_left_px'][1])
        self.n_pixel_vert = (self.bounding_box['lower_right_px'][0]
                             - self.bounding_box['upper_left_px'][0])
        assert self.n_pixel_horz > 0
        assert self.n_pixel_vert > 0

        self.aspect_ratio = float(self.n_pixel_horz) / self.n_pixel_vert
        if self.aspect_ratio > 1:
            self.is_wide = True
            self.long_edge = self.n_pixel_horz  # Length of edge in pixels
            self.short_edge = self.n_pixel_vert  # Length of edge in pixels
        else:
            self.is_wide = False
            self.short_edge = self.n_pixel_horz  # Length of edge in pixels
            self.long_edge = self.n_pixel_vert  # Length of edge in pixels
            self.aspect_ratio = 1. / self.aspect_ratio  # Make aspect ratio >= 1

        self.start_pt_min_px_dist_from_long_edge = int(self.START_PT_MIN_DIST_FROM_LONG_EDGE
                                                       // self.resolution)
        self.start_pt_min_px_dist_from_short_edge = int(self.START_PT_MIN_DIST_FROM_SHORT_EDGE
                                                        // self.resolution)

        # Important for self.sample_once()
        is_valid_1 = (self.start_pt_min_px_dist_from_long_edge
                      < (self.short_edge - self.start_pt_min_px_dist_from_long_edge))
        is_valid_2 = (self.start_pt_min_px_dist_from_short_edge
                      < (self.long_edge - self.start_pt_min_px_dist_from_short_edge))

        self.is_valid = is_valid_1 and is_valid_2

    def unnormalize_px_coord(self, normalized_coord):
        """Unnormalize the pixel coordinates. Most computation is done in normalized coordinates
        such that the top-left corner of the bounding box is [0, 0] and the long edge is horizontal.
        Unnormalizing the coordinates simply translates, and possibly reflects/flips, the bounding
        box.

        Normalized coordinates:

        (0, 0)                                   (self.long_edge - 1, 0)
        ---------------------------------------------
        |                                           |
        |                                           |
        |                                           |
        |                                           |
        |                                           |
        ---------------------------------------------
        (self.short_edge - 1, 0)                 (self.short_edge -1, self.long_edge - 1)


        Unnormalized box:
        (0, 0)
        --------------
        |            |
        |            |
        |            |
        |            |
        |            |
        |            |
        |            |
        |            |
        |            |
        |            |
        |            |
        |            |
        |            |
        |            |
        --------------  (self.long_edge, self.short_edge)

        Args:
            normalized_coord: Numpy array of shape (2,) representing normalized pixel coordinates

        Returns:
            unnormalized_coord: Numpy array of shape (2,) representing coordinates of query pixel
                    in the map.
        """
        normalized_coord = np.asarray(normalized_coord)
        if not self.is_wide:
            normalized_coord = normalized_coord[::-1]
        return self.bounding_box['upper_left_px'] + normalized_coord

    def is_valid_point(self, query_px_coord):
        """Check if the point is within the corridor (not free space or another room), and check
        that point is not on (or close to) clutter.

        Args:
            query_px_coord: Location of query point in pixel coordinates.

        Returns:
            is_valid: Whether queried location is valid.
        """
        return ((self.semantic_map[query_px_coord[0], query_px_coord[1]] == self.id)
                and (self.clutter_map[query_px_coord[0], query_px_coord[1]] == 0))

    def sample_once(self):
        """Sample a (start) point in the room. Returns None if the current room is invalid.

        Steps:
            1. Sample a start point anywhere in the room, as long as it is a minimum distance
               (MIN_DIST_FROM_SHORT_EDGE) from the short edge.
            2. Make sure the sampled point is valid. Repeat until it is.

        Returns:
            traj_start_end: Dict containing start and end points of trajectory in pixel coordinates.
        """
        if not self.is_valid:
            return None

        found_valid_sample = False
        max_iterations = 50
        cur_iteration = 0
        while not found_valid_sample:
            row = np.random.randint(self.start_pt_min_px_dist_from_long_edge,
                                    self.short_edge - self.start_pt_min_px_dist_from_long_edge)
            col = np.random.randint(self.start_pt_min_px_dist_from_short_edge,
                                    self.long_edge - self.start_pt_min_px_dist_from_short_edge)
            start_point = self.unnormalize_px_coord([row, col])
            found_valid_sample = self.is_valid_point(start_point)

            cur_iteration += 1
            if cur_iteration == max_iterations:
                return None

        return start_point
