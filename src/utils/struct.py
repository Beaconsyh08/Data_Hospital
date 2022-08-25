#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   struct.py
@Time    :   2022/04/08 18:09:44
@Author  :   chenminghua 
'''
import pandas as pd

class Obstacle(object):
    """
    Description: define obstacle attribute
    """
    def __init__(self) -> None:
        self.img_path = None
        self.json_path = None
        self.id = None
        self.dtgt = None
        self.flag = None # good/miss/false
        self.class_name = None
        self.bbox = []  # [x,y,w,h]
        self.truncation = None
        self.crowding = None
        self.occlusion = None
        self.x = None
        self.y = None
        self.z = None
        self.position = [] #[x,y,z]
        self.height = None
        self.length = None
        self.width = None
        self.yaw = None
        self.camera_orientation = None
        self.flag = None
        self.case_flag = None


def parse_obs(row):
    # if type(row) != pd.Series:
    #     self.logger.warning("type(row) != pd.Series, parse obs error")
    #     return
    obs = Obstacle()
    try:
        obs.img_path = row.name.split('@')[0]
    except Exception:
        try:
            obs.img_path = row.Index.split('@')[0]
        except:
            try:
                obs.img_path = row.index.split('@')[0]
            except:
                print(row)
            
    obs.id = row.id
    try:
        obs.dtgt = 'gt' if pd.isna(row.dtgt) else row.dtgt
    except:
        obs.dtgt = "gt"

    obs.bbox = [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h]
    obs.flag = row.flag
    obs.position = [row.x, row.y, row.z]
    obs.x, obs.y, obs.z = obs.position
    obs.height, obs.length, obs.width = row.height, row.length, row.width
    obs.class_name, obs.yaw = row.class_name, row.yaw
    try:
        obs.truncation, obs.crowding, obs.occlusion = row.truncation, row.crowding, row.occlusion
    except:
        pass
    obs.camera_orientation = row.camera_orientation
    obs.flag = row.flag
    obs.json_path = row.json_path
    
    try:
        obs.case_flag = row.case_flag
    except:
        obs.case_flag = 0
        
    return obs