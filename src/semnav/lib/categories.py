from enum import IntEnum, unique


@unique
class SemanticCategory(IntEnum):
    """Semantic classes.
    """
    UNK = 0
    ceiling = 1
    floor = 2
    window = 3
    door = 4
    column = 5
    beam = 6
    wall = 7
    sofa = 8
    chair = 9
    table = 10
    board = 11
    bookcase = 12
    clutter = 13


@unique
class NavPlanDifficulty(IntEnum):
    """Difficulty of navigation plan.
    """
    easy = 0
    moderate = 1
    hard = 2


@unique
class BehaviorCategory(IntEnum):
    """Behavior ID/edge in the semantic graph.
    """
    fd = 0
    tl = 1
    tr = 2
    cf = 3
    s = 4


affordance_types = [behavior_id.name for behavior_id in BehaviorCategory]


@unique
class RoomCategory(IntEnum):
    """Category of each room/node in the semantic graph.
    """
    room = 0
    hallway = 1
    open_space = 2


room2category_dict = {
    'auditorium': RoomCategory.room,
    'conferenceRoom': RoomCategory.room,
    'copyRoom': RoomCategory.room,
    'hallway': RoomCategory.hallway,
    'lobby': RoomCategory.hallway,
    'lounge': RoomCategory.open_space,
    'office': RoomCategory.room,
    'openspace': RoomCategory.open_space,
    'pantry': RoomCategory.room,
    'storage': RoomCategory.room,
    'WC': RoomCategory.room,
    }


def room2category(room_name):
    for key, room_category in room2category_dict.iteritems():
        if room_name.startswith(key):
            return room_category
    raise ValueError('No category for this type of room: {}'.format(room_name))


def behavior_id2category(behavior_id):
    for name, member in BehaviorCategory.__members__.items():
        if behavior_id.startswith(name):
            return member
    raise ValueError('No category for this behavior ID: {}'.format(behavior_id))
