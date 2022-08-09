

# 毫末视觉障碍物类别
CLASSES = ["car", "bus", "truck", "rider", "tricycle", "bicycle", "pedestrian"]
CLASS_MAPS = dict(
    car=['car'],
    bus=['bus'],
    truck=['truck'],
    rider=['rider'],
    tricycle=['tricycle'],
    bicycle=['bicycle'],
    pedestrian=['pedestrian']
)

# 车辆的方向
DIRECTIONS = ['left', 'right', 'front', 'behind', 'left_front', 'left_behind', 'right_front', 'right_behind']
DIRECTIONS_MAPS = dict(
        left=['left'],
        right=['right'],
        front=['front'],
        behind=['behind', 'behin'],
        left_front=['left_front', 'left front'],
        left_behind = ['left_behind', 'left behind'],
        right_front = ['right_front', 'right front'],
        right_behind = ['right_behind', 'right behind']
)

DEFAULT_ATTR_ID = 0

def transfer_category(category):
    category_id = None
    for key, value in CLASS_MAPS.items():
        if category in value:
            category_id = CLASSES.index(key) + 1
    return category_id

def transfer_direction(direction):
    """
    车辆方向默认为 Left(0)
    """
    direction_id = DEFAULT_ATTR_ID
    if direction is None:
        return direction_id

    for key, value in DIRECTIONS_MAPS.items():
        if direction.lower() in value:
            direction_id = DIRECTIONS.index(key)
    return direction_id


def transfer_attr(value):
    """
    属性默认为 0
    """
    attr_id = DEFAULT_ATTR_ID
    if value is not None:
        if isinstance(value, str):
            try:
                return int(value.strip())
            except:
                return attr_id
        elif isinstance(value, int):
            return value
    return attr_id