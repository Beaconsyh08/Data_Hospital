
from .attrs_parser import transfer_direction

class Fake3DParser(object):
    def __init__(self) -> None:
        super().__init__()

    def parse_edges(self, obj_ann):
        attrs = obj_ann.get("attributes")
        vertical_edge = attrs.get("side_edge", None)
        bottom_edge = attrs.get("bottom_edge", None)
        return vertical_edge, bottom_edge

    def _get_y_pos(self, x, slope):
        x_diff = slope[0] - slope[2]
        y_diff = slope[1] - slope[3]

        if abs(x_diff) < 5:
            k = 0
        else:
            k = y_diff / x_diff
        b = slope[1] - k * slope[0]
        return k * x + b
    
    def _get_x_pos(self, y, slope):
        x_diff = slope[0] - slope[2]
        y_diff = slope[1] - slope[3]
        if abs(y_diff) < 5:
            k = 0
        else:
            k = x_diff / y_diff
        b = slope[0] - k * slope[1]
        return k * y + b

    def get_y_pos(self, x, slope):
        y = self._get_y_pos(x, slope)
        if self.image_height is not None and y >= self.image_height:
            x2h = self._get_x_pos(self.image_height - 1, slope)
            x = x2h
            y = self.image_height - 1
        return x, y

    def parse_left(self, obj_ann):
        xmin, ymin, w, h = obj_ann['bbox']
        xmax = xmin + w
        ymax = ymin + h
        vertical_edge, bottom_edge = self.parse_edges(obj_ann)
        # 单方向不应该存在vertical edge
        if vertical_edge is not None:
            return None

        # 单方向可能存在bottom edge
        if bottom_edge is None:
            slopes = [xmin, ymax, xmax, ymax]
        else:
            start_point = bottom_edge.get('start_point', None)
            end_point = bottom_edge.get('end_point', None)
            # 如果start_point, end_point有一个为None，则认为标注错误
            if start_point is None or end_point is None:
                return None
            slopes = start_point + end_point
        
        lf_x = min(xmin, xmax)
        lb_x = max(xmin, xmax)
        lf_x, lf_y = self.get_y_pos(lf_x, slopes)
        lb_x, lb_y = self.get_y_pos(lb_x, slopes)
        side_edge = [lf_x, lf_y, lb_x, lb_y]
        return [[lf_x, lf_y], [lb_x, lb_y], None, None, side_edge]

    def parse_right(self, obj_ann):
        xmin, ymin, w, h = obj_ann['bbox']
        xmax = xmin + w
        ymax = ymin + h
        vertical_edge, bottom_edge = self.parse_edges(obj_ann)
        # 单方向不应该存在vertical edge
        if vertical_edge is not None:
            return None
        
        # 单方向可能存在bottom edge
        if bottom_edge is None:
            slopes = [xmin, ymax, xmax, ymax]
        else:
            start_point = bottom_edge.get('start_point', None)
            end_point = bottom_edge.get('end_point', None)
            # 如果start_point, end_point有一个为None，则认为标注错误
            if start_point is None or end_point is None:
                return None
            slopes = start_point + end_point

        rf_x = max(xmin, xmax)
        rb_x = min(xmin, xmax)
        rf_x, rf_y = self.get_y_pos(rf_x, slopes)
        rb_x, rb_y = self.get_y_pos(rb_x, slopes)
        side_edge = [rf_x, rf_y, rb_x, rb_y]
        return [None, None, [rf_x, rf_y], [rb_x, rb_y], side_edge]        


    def parse_front(self, obj_ann):
        xmin, ymin, w, h = obj_ann['bbox']
        xmax = xmin + w
        ymax = ymin + h
        vertical_edge, bottom_edge = self.parse_edges(obj_ann)
        # 单方向不应该存在vertical edge
        if vertical_edge is not None:
            return None

        # 单方向不应该存在bottom edge
        if bottom_edge is not None:
            return None

        lf_x = max(xmin, xmax)
        rf_x = min(xmin, xmax)
        lf_y = ymax
        rf_y = ymax
                
        return [[lf_x, lf_y], None, [rf_x, rf_y], None, None]

    def parse_behind(self, obj_ann):
        xmin, ymin, w, h = obj_ann['bbox']
        xmax = xmin + w
        ymax = ymin + h
        vertical_edge, bottom_edge = self.parse_edges(obj_ann)
        # 单方向不应该存在vertical edge
        if vertical_edge is not None:
            return None

        # 单方向不应该存在bottom edge
        if bottom_edge is not None:
            return None
        
        lb_x = min(xmin, xmax)
        rb_x = max(xmin, xmax)
        lb_y = ymax
        rb_y = ymax

        return [None, [lb_x, lb_y], None, [rb_x, rb_y], None]

    def parse_left_behind(self, obj_ann):
        xmin, ymin, w, h = obj_ann['bbox']
        xmax = xmin + w
        ymax = ymin + h
        vertical_edge, bottom_edge = self.parse_edges(obj_ann)
        
        # left behind 目标必须存在vertical edge 和 bottom edge
        if vertical_edge is None or bottom_edge is None:
            return None

        start_point = bottom_edge.get('start_point', None)
        end_point = bottom_edge.get('end_point', None)
        # 如果start_point, end_point有一个为None，则认为标注错误
        if start_point is None or end_point is None:
            return None
        slopes = start_point + end_point

        lf_x = min(xmin, xmax)
        lb_x = vertical_edge
        rb_x = max(xmin, xmax)
        lf_x, lf_y = self.get_y_pos(lf_x, slopes)
        lb_x, lb_y = self.get_y_pos(lb_x, slopes)
        rb_y = ymax
        side_edge = [lf_x, lf_y, lb_x, lb_y]
        return [[lf_x, lf_y], [lb_x, lb_y], None, [rb_x, rb_y], side_edge]
        
    def parse_right_behind(self, obj_ann):
        xmin, ymin, w, h = obj_ann['bbox']
        xmax = xmin + w
        ymax = ymin + h
        vertical_edge, bottom_edge = self.parse_edges(obj_ann)
        
        # left behind 目标必须存在vertical edge 和 bottom edge
        if vertical_edge is None or bottom_edge is None:
            return None

        start_point = bottom_edge.get('start_point', None)
        end_point = bottom_edge.get('end_point', None)
        # 如果start_point, end_point有一个为None，则认为标注错误
        if start_point is None or end_point is None:
            return None
        slopes = start_point + end_point

        lb_x = min(xmin, xmax)
        rb_x = vertical_edge
        rf_x = max(xmin, xmax)

        lb_y = ymax
        rb_x, rb_y = self.get_y_pos(rb_x, slopes)
        rf_x, rf_y = self.get_y_pos(rf_x, slopes)
        side_edge = [rf_x, rf_y, rb_x, rb_y]
        return [None, [lb_x, lb_y], [rf_x, rf_y], [rb_x, rb_y], side_edge]        

    def parse_left_front(self, obj_ann):
        xmin, ymin, w, h = obj_ann['bbox']
        xmax = xmin + w
        ymax = ymin + h
        vertical_edge, bottom_edge = self.parse_edges(obj_ann)
        # left behind 目标必须存在vertical edge 和 bottom edge
        if vertical_edge is None or bottom_edge is None:
            return None

        start_point = bottom_edge.get('start_point', None)
        end_point = bottom_edge.get('end_point', None)
        # 如果start_point, end_point有一个为None，则认为标注错误
        if start_point is None or end_point is None:
            return None
        slopes = start_point + end_point

        rf_x = min(xmin, xmax)
        lf_x = vertical_edge
        lb_x = max(xmin, xmax)

        rf_y = ymax
        lf_x, lf_y = self.get_y_pos(lf_x, slopes)
        lb_x, lb_y = self.get_y_pos(lb_x, slopes)
        side_edge = [lf_x, lf_y, lb_x, lb_y]
        return [[lf_x, lf_y], [lb_x, lb_y], [rf_x, rf_y], None, side_edge]        

    def parse_right_front(self, obj_ann):
        xmin, ymin, w, h = obj_ann['bbox']
        xmax = xmin + w
        ymax = ymin + h
        vertical_edge, bottom_edge = self.parse_edges(obj_ann)
        
        # left behind 目标必须存在vertical edge 和 bottom edge
        if vertical_edge is None or bottom_edge is None:
            return None

        start_point = bottom_edge.get('start_point', None)
        end_point = bottom_edge.get('end_point', None)
        # 如果start_point, end_point有一个为None，则认为标注错误
        if start_point is None or end_point is None:
            return None
        slopes = start_point + end_point

        rb_x = min(xmin, xmax)
        rf_x = vertical_edge
        lf_x = max(xmin, xmax)

        rb_x, rb_y = self.get_y_pos(rb_x, slopes)
        rf_x, rf_y = self.get_y_pos(rf_x, slopes)
        lf_y = ymax        
        side_edge = [rf_x, rf_y, rb_x, rb_y]
        return [[lf_x, lf_y], None, [rf_x, rf_y], [rb_x, rb_y], side_edge]

    def __call__(self, obj_ann: dict, image_width=None, image_height=None):
        self.image_width = image_width
        self.image_height = image_height
        # 如果没有attributes字段，返回None
        attrs = obj_ann.get('attributes', None)
        if attrs is None:
            return None
        
        # 如果没有direction字段，返回None
        direction = attrs.get('direction', None)
        if direction is None:
            return None
        
        # 如果direction字段内容非法，返回None
        direction_id = transfer_direction(direction)
        if direction_id == -1:
            return None

        if 0 == direction_id:
            # Left
            vedges = self.parse_left(obj_ann)
        elif 1 == direction_id:
            # Right
            vedges = self.parse_right(obj_ann)
        elif 2 == direction_id:
            # Front
            vedges = self.parse_front(obj_ann)
        elif 3 == direction_id:
            # Behind
            vedges = self.parse_behind(obj_ann)
        elif 4 == direction_id:
            # Left Front
            vedges = self.parse_left_front(obj_ann)
        elif 5 == direction_id:
            # Left Behind
            vedges = self.parse_left_behind(obj_ann)
        elif 6 == direction_id:
            # Right Front
            vedges = self.parse_right_front(obj_ann)
        elif 7 == direction_id:
            # Right Behind
            vedges = self.parse_right_behind(obj_ann)
        else:
            vedges = None

        return vedges