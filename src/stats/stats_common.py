#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   stats_common.py
@Time    :   2022/04/07 11:59:53
@Author  :   chenminghua 
'''

import pandas as pd
from shapely.geometry import Point, Polygon
from typing import List, Dict, Tuple
from src.utils.struct import Obstacle
TYPE_MAP = {'car': 'car', 'van': 'car', 
            'truck': 'truck', 'forklift': 'truck',
            'bus':'bus', 
            'rider':'rider',
            'rider-bicycle': 'rider', 'rider-motorcycle':'rider', 
            'rider_bicycle': 'rider', 'rider_motorcycle':'rider',
            'bicycle': 'bicycle', 'motorcycle': 'bicycle',
            'tricycle': 'tricycle', 'closed-tricycle':'tricycle', 'open-tricycle': 'tricycle', 
            'closed_tricycle':'tricycle', 'open_tricycle': 'tricycle', 'pedestrian': 'pedestrian',
            }


category = {
    'vehicle': ['car', 'bus', 'truck', 'van', 'forklift'],
    'vru': ['pedestrian', 'rider', 'bicycle', 'tricycle', 'rider-bicycle', 'rider-motorcycle', 'rider_bicycle', 'rider_motorcycle', 'closed-tricycle', 'open-tricycle', 'closed_tricycle', 'open_tricycle'],
    'static': ['trafficCone', 'water-filledBarrier', 'other', 'accident', 'construction']
}
truncation_area = [[20,2], [20,-2], [-20, -2], [-20, 2]]

def define_category(df):
    """
    define obstacles, including vehicle, vru, static
    """
    df['broad_category'] = None
    for key, values in category.items():
        df['broad_category'][df['class_name'].isin(values)] = key
    return df

truncation_area = [[20,2], [20,-2], [-20, -2], [-20, 2]]
def check_truncation(obstacle: Obstacle):  
    if obstacle.bbox == None or obstacle.position == None:
        return False
    bbox_x, bbox_y, bbox_w, bbox_h = obstacle.bbox
    x, y = obstacle.x, obstacle.y
    ploy = Polygon(truncation_area)
    point = Point([x, y])
    if ploy.contains(point):
        return "truncation_front_rear"
    if bbox_x < 10 or bbox_x + bbox_w > 1910:
        return "truncation_left_right"
    if pd.isna(obstacle.truncation) or obstacle == None:
        return False
    if int(obstacle.truncation) == 1:
        if bbox_x < 30 or bbox_x + bbox_w > 1900: 
            return "truncation_left_right"
    return False

def check_bbox_size(bbox:List[float] = None):
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    if bbox_w * bbox_h > 1920*1080*0.5:
        return "bbox_large"
    return False

def check_truncation_from_bbox(obstacle: Obstacle):  
    if obstacle.bbox == None or len(obstacle.bbox) != 4:
        return False
    bbox_x, bbox_y, bbox_w, bbox_h = obstacle.bbox

    if bbox_x < 10 or bbox_x + bbox_w > 1910:
        return True

    if bbox_x < 30 or bbox_x + bbox_w > 1900: 
        return True

    return False

truncation_area_train = [[2,-20], [-2,-20], [-2, 20], [2, 20]]
def check_truncation_train(obstacle: Obstacle):  
    if obstacle.bbox == None or obstacle.position == None:
        return False
    bbox_x, bbox_y, bbox_w, bbox_h = obstacle.bbox
    x, y = obstacle.x, obstacle.y
    ploy = Polygon(truncation_area_train)
    point = Point([x, y])
    if ploy.contains(point):
        return "truncation_front_rear"
    if bbox_x < 10 or bbox_x + bbox_w > 1910:
        return "truncation_left_right"
    if pd.isna(obstacle.truncation) or obstacle == None:
        return False
    if int(obstacle.truncation) == 1:
        if bbox_x < 30 or bbox_x + bbox_w > 1900: 
            return "truncation_left_right"
    return False