#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tagging.py
@Time    :   2022/05/25 20:02:41
@Author  :   chenminghua 
'''
import datetime
import math
import pandas as  pd
from configs.config import Config
from src.utils.logger import get_logger
import src.data_manager.data_manager as dm
from src.stats.stats_common import check_truncation
from src.utils.struct import parse_obs

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

class Tagging(object):
    def __init__(self) -> None:
        self.config = Config
        self.df_path = self.config.DATAFRAME_PATH
    

    def _load_data(self, ):
        self.df = dm.load_from_pickle(self.df_path)
        # pass

    def process(self, ) -> None:
        # load datasets
        self._load_data()
        self.df = self.df[self.df.priority == 'P0']
        self.df = self.df[~self.df.index.duplicated(keep='first')]

        # obstacle type
        # self.df['class_name'] = self.df['class_name'].map(TYPE_MAP)
        self.df['tag_type'] = self.df['class_name'].map(TYPE_MAP)
        # self.df['class_name'] = self.df['class_name'].map(TYPE_MAP)
        self.df = self.df[~self.df['class_name'].isna()]

        # obstacle velocity
        # TODO

        # obstacle heading
        self.df.loc[(self.df.yaw > math.pi), 'yaw'] = self.df.loc[(self.df.yaw > math.pi)].yaw - math.pi
        self.df['tag_heading'] = None
        self.df['yaw'] *= 180 / math.pi
        yaw = self.df.yaw
        self.df.loc[((yaw > -45) & (yaw < 45)), 'tag_heading'] = 'forward'
        self.df.loc[((yaw < -135 ) | (yaw > 135)), 'tag_heading'] = 'reverse'
        self.df.loc[(((yaw < -45)& (yaw > -135)) | ((yaw > 45) & (yaw < 135))), 'tag_heading'] = 'transverse'

        # obstacle distance
        self.df['tag_distance'] = None
        self.df.loc[self.df['self_dis'].between(0, 5), 'tag_distance'] = '0-5'
        self.df.loc[self.df['self_dis'].between(5, 10),'tag_distance' ] = '5-10'
        self.df.loc[self.df['self_dis'].between(10, 20), 'tag_distance'] = '10-20'
        self.df.loc[self.df['self_dis'].between(20, 60), 'tag_distance'] = '20-60'
        self.df.loc[self.df['self_dis'].between(60, 200), 'tag_distance'] = '60-200'

        # truncation
        for row in self.df.itertuples():
            obs = parse_obs(row)
            state = check_truncation(obs)
            if state:
                self.df.at[row.Index, 'tag_truncation'] = state
            else:
                self.df.at[row.Index, 'tag_truncation'] = 'no_truncation'
        
        # obstacle position
        self.df['tag_area'] =  None
        self.df.loc[(self.df.x > 5) & (self.df.y > 1), 'tag_area'] = 'front_left'
        self.df.loc[(self.df.x > 5) & (abs(self.df.y) < 1), 'tag_area'] = 'front'
        self.df.loc[(self.df.x > 5) & (self.df.y < -1), 'tag_area'] = 'front_right'
        self.df.loc[(self.df.x > 0) & (self.df.x < 5) & (self.df.y > 0), 'tag_area'] = 'left'
        self.df.loc[(self.df.x > 0) & (self.df.x < 5) & (self.df.y < 0), 'tag_area'] = 'right'
        self.df.loc[(self.df.x < 0) & (self.df.y > 1), 'tag_area'] = 'rear_left'
        self.df.loc[(self.df.x < 0) & (abs(self.df.y) < 1), 'tag_area'] = 'rear'
        self.df.loc[(self.df.x < 0) & (self.df.y < -1), 'tag_area'] = 'rear_right'

        # time
        self.df['tag_time'] = self.df['time'].apply(lambda x: 'night' if x >=datetime.time(19,00,00) or x <=datetime.time(7,00,00) else 'daytime')

        # save dataframe
        self.save_dataframe()
    
    def save_dataframe(self, ):
        temp_dm = dm.DataManager(self.df)
        temp_dm.save_to_pickle(self.df_path) 

        

        




