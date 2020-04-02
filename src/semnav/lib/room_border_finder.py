from __future__ import print_function
from __future__ import division

import numpy as np
import os
import yaml

from enum import Enum, unique  # , auto
from semnav.lib.utils import read_area_data


@unique
class RoomAdjRelationship(Enum):
    """Relationship enumeration between two possibly adjacent rooms.
    """
    LEFT = 1  # auto()
    RIGHT = 2  # auto()
    ABOVE = 3  # auto()
    BELOW = 4  # auto()
    NOT_ADJACENT = 5  # auto()


class RoomBorderFinder(object):
    """Given two rooms, find where they border each other. Assume the border is axis-aligned.
    """

    def __init__(self, yaml_filepath, adjacent_edge_threshold=1.):
        # Load yaml file containing map data
        with open(yaml_filepath, 'r') as f:
            map_yaml = yaml.load(f)

        mat_filepath = os.path.join(os.path.dirname(yaml_filepath), map_yaml['mat_filepath'])
        self.area_data = read_area_data(mat_filepath)

        # Maximum distance (meters) for which edges of two rooms would be considered adjacent
        # Convert from meters to number of pixels
        self.adjacent_edge_threshold = adjacent_edge_threshold / self.area_data['resolution']

        self.rooms = {room_dict['name']: room_dict for room_dict in self.area_data['rooms']}

    def get_room_dict(self, room_name):
        """Return the room dict for a room of the given name.
        """
        if room_name in self.rooms:
            return self.rooms[room_name]
        else:
            raise ValueError('Invalid room name.')

    def get_border(self, room_name_1, room_name_2):
        """Get the border between the two rooms. If they are not adjacent, return (None, None). Note
        that two rooms (e.g. corridor and room) can be considered as adjacent even if there is no
        door connecting them.

        Args:
            room_name_1: Name of room 1. The bounding box edges of this room will be returned if the
                two rooms are found to be adjacent.
            room_name_2: Name of room 2.

        Returns:
            row_val: A np.int32 (pixel coordinates) representing the boundary between two rooms.
                This is based off selecting the correct edge from the bounding box of room 1.
            col_val: A np.int32 (pixel coordinates) representing the boundary between two rooms.
                This is based off selecting the correct edge from the bounding box of room 1.
            relationship: RoomAdjRelationship between room 1 and room 2 in the following manner:
                    room 1 is *relationship* (of) room 2. Example: room 1 is RIGHT of room 2.
        """
        room_dict1, room_dict2 = (self.get_room_dict(room_name_1), self.get_room_dict(room_name_2))
        bb1, bb2 = (room_dict1['bounding_box'], room_dict2['bounding_box'])

        row_val, col_val = (None, None)
        dists = np.asarray([
            np.abs(bb1['upper_left_px'][0] - bb2['lower_right_px'][0]),
            np.abs(bb1['lower_right_px'][0] - bb2['upper_left_px'][0]),
            np.abs(bb1['upper_left_px'][1] - bb2['lower_right_px'][1]),
            np.abs(bb1['lower_right_px'][1] - bb2['upper_left_px'][1]),
            ])
        min_index = np.argmin(dists)
        if dists[min_index] <= self.adjacent_edge_threshold:
            print('room 1:', room_name_1)
            print('room 2:', room_name_2)
            # Compare top edge of bb1 with bottom edge of bb2
            if min_index == 0:
                print('below')
                row_val = bb1['upper_left_px'][0]
                relationship = RoomAdjRelationship.BELOW
            # Compare bottom edge of bb1 with top edge of bb2
            elif min_index == 1:
                print('above')
                row_val = bb1['lower_right_px'][0]
                relationship = RoomAdjRelationship.ABOVE
            # Compare left edge of bb1 with right edge of bb2
            elif min_index == 2:
                print('right')
                col_val = bb1['upper_left_px'][1]
                relationship = RoomAdjRelationship.RIGHT
            # Compare right edge of bb1 with left edge of bb2
            elif min_index == 3:
                print('left')
                col_val = bb1['lower_right_px'][1]
                relationship = RoomAdjRelationship.LEFT
            else:
                raise ValueError
        else:  # Not adjacent
            relationship = RoomAdjRelationship.NOT_ADJACENT
        return row_val, col_val, relationship
