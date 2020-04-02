"""Sample start and end locations within a corridor.
"""

# from __future__ import print_function
from __future__ import division

import numpy as np
import random

from semnav.lib.room_sampler import RoomSampler


class CorridorSampler(RoomSampler):
    """Sample start (spawn) and end points in a corridor.
    """

    LONG_EDGE_MIN = 4.  # Meters
    MIN_SAMPLING_ASPECT_RATIO = 2.
    END_PT_DIST_FROM_SHORT_EDGE = 1  # Meters

    def __init__(self, semantic_map, clutter_map, room_dict, resolution):
        super(CorridorSampler, self).__init__(semantic_map, clutter_map, room_dict, resolution)
        assert room_dict['name'].startswith('hallway')

        # It makes sense to sample only if the aspect ratio is above a minimum
        # Also, the corridor has to be sufficiently long/large.
        self.is_valid = (self.is_valid
                         and (self.aspect_ratio > self.MIN_SAMPLING_ASPECT_RATIO)
                         and ((self.long_edge * self.resolution) > self.LONG_EDGE_MIN))

        self.end_point_px_dist_from_short_edge = int(self.END_PT_DIST_FROM_SHORT_EDGE
                                                     // self.resolution)

        # The end points are fixed to two particular locations
        # Order of points if self.is_wide is True: [left, right]
        # Order of points if self.is_wide is False: [top, bottom]
        row = self.short_edge // 2
        col_left = self.end_point_px_dist_from_short_edge
        col_right = self.long_edge - self.end_point_px_dist_from_short_edge
        normalized_end_points = ([row, col_left], [row, col_right])
        self.end_points = [self.unnormalize_px_coord(normalized_end_points[0]),
                           self.unnormalize_px_coord(normalized_end_points[1])]
        for idx, (end_pt, normalized_end_pt) in enumerate(zip(self.end_points,
                                                              normalized_end_points)):
            if not self.is_valid_point(end_pt):
                self.end_points[idx] = self.find_close_valid_point(normalized_end_pt)

    def find_close_valid_point(self, normalized_px_coord):
        """Find a nearby valid point by searching in short-edge direction.

        Args:
            normalized_px_coord: Pixel coordinates of query point.

        Returns:
            px_coord: Normalized pixel coordinates of nearest point.
        """
        found_valid_point = False
        dist_from_query = 0
        while not found_valid_point:
            dist_from_query += 1
            px_coord_above = [normalized_px_coord[0] - dist_from_query, normalized_px_coord[1]]
            px_coord_below = [normalized_px_coord[0] + dist_from_query, normalized_px_coord[1]]

            if self.is_valid_point(self.unnormalize_px_coord(px_coord_above)):
                found_valid_point = True
                return self.unnormalize_px_coord(px_coord_above)
            elif self.is_valid_point(self.unnormalize_px_coord(px_coord_below)):
                found_valid_point = True
                return self.unnormalize_px_coord(px_coord_below)

            if (px_coord_above[0] == 0) or (px_coord_below[0] == (self.short_edge - 1)):
                raise AssertionError('Cannot find good corridor endpoint.')

    def sample_end_pt(self):
        """Sample an end point in the corridor (located at one end of the corridor or another).
        """
        end_point = random.choice(self.end_points)  # Uniformly choose an end point
        return end_point
