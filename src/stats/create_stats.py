#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   create_stats.py
@Time    :   2022/01/03 16:36:08
@Author  :   chenminghua 
'''

import json
import sys
from configs.config import StatsConfig
from src.stats.stats import Stats
from src.stats.stats_cam2d import StatsCam2D
from src.stats.stats_cam3d import StatsCam3D
from src.stats.stats_lidar import StatsLidar
from src.utils.logger import get_logger


class StatsCreator():
    def __init__(self, config=None) -> None:
        self.config = config if config is not None else StatsConfig
        self.create_stats()   

    def create_stats(self, ):
        stats_type = self.config.STATS_TYPE
        if self.config.STATS_TYPE == 'cam2d':
            self.stats_manager = StatsCam2D(config = self.config)
        elif self.config.STATS_TYPE == 'cam3d':
            self.stats_manager = StatsCam3D(config = self.config)
        elif self.config.STATS_TYPE == 'lidar':
            self.stats_manager = StatsLidar(config = self.config)
    
    def run(self, ):
        self.stats_manager.run()