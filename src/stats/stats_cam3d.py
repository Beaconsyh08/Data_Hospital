#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   stats.py
@Time    :   2021/11/15 20:17:09
@Email   :   songyuhao@haomo.ai
@Author  :   YUHAO SONG 

Copyright (c) HAOMO.AI, Inc. and its affiliates. All Rights Reserved
"""
import collections
import copy
import math
import os
import random
import threading
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import src.data_manager.data_manager as dm
from configs.config import *
from configs.config import (Config, OutputConfig, StatsConfig,
                            VisualizationConfig)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from src.classification.clustering import Clustering
from src.stats.base_stats import BaseStats
from src.stats.metrics import Metrics
from src.stats.multi_stats import MultiStats
from src.stats.error_analysis import ErrorAnalysis
from src.stats.stats_common import *
from src.utils.common import compute_iou
from src.utils.file_io import write_json
from src.utils.logger import get_logger
from src.utils.struct import Obstacle, parse_obs
from src.visualization.visualization import Visualizer
from tqdm import tqdm


class StatsCam3D(BaseStats):

    def __init__(self, df: pd.DataFrame = None, df_path: str = None, 
                        emb_path: str = None, config: dict = None) -> None:
        
        super(StatsCam3D, self).__init__(config=config, df=df,
                                        df_path=df_path,emb_path=emb_path)
        self.indicator_panel = {}
        self.save_dir = Path(OutputConfig.OTHERS_OUTPUT_DIR)
        self.indicator_path = self.save_dir / 'indicator_panel.json'
        self.evaluation = Metrics()
        self.error_analysis = ErrorAnalysis()
        self.multi_version = MultiStats()
        self.multi_result = None
        self.metrics = None

    
    def run(self, ):
        # load data
        BaseStats.run(self)
        # self._define_priority()
        # compute metrics and error analysis
        self.overall_metric()
        self.evaluation_results_analysis()

        # multi version comparison
        self.compare_multi_version()

        # output 
        self.output_result()
        
        temp_dm = dm.DataManager(self.df)
        temp_dm.save_to_pickle(self.df_path)        
    

    def overall_metric(self, ):
        """
        Description: computing precision, recall, f1-score for overall/P0/P1
        """
        self.evaluation.load_data(self.df)
        self.metrics = self.evaluation.compute_metrics()
        self.evaluation.show_metrics()

    def compare_multi_version(self, ):
        """
        Description: Comparison of multi version evaluation results
        """
        self.multi_version.load_data(self.df_multi_version)
        df_multi= self.multi_version.define_changes()
        df_multi_p0 = df_multi[df_multi['priority'] == 'P0']
        print("df_multi_p0", df_multi_p0)
        self.multi_result = self.multi_version.compute_metrics(df_multi_p0)
        self.multi_version.show_multi_version_cases(df_multi_p0)
        
    # def _define_priority(self, ):
    #     """
    #     Description: Redefine priority level P0/P1/P2
    #     Param: 
    #     Returns: 
    #     """
    #     self.logger.info('Redefine P0/P1/P2')
        
    #     for row in tqdm(self.df.itertuples(), total=len(self.df)):
    #         class_name, flag, occlusion, crowding = row.class_name, row.flag, row.occlusion, row.crowding
    #         pos_x, pos_y = row.x, row.y
            
    #         occlusion = 0 if math.isnan(occlusion) else occlusion
    #         crowding = 0 if math.isnan(crowding) else crowding
    #         priority = 'P0'
    #         if row.euclidean_dis_diff < 0.3:
    #             priority = 'P1'
    #         if abs(pos_x) > 20 or abs(pos_y) > 10 or abs(pos_y)  < 1:
    #             priority = 'P1'
    #         if (occlusion != None and int(occlusion) > 0) or (crowding != None and int(crowding) > 0):
    #             priority = 'P1'

    #         self.df.at[row.Index, 'priority'] = priority 
    
    def evaluation_results_analysis(self, ):
        """
        Description: false_positive_analysis and false_negative_analysis
        """
        self.error_analysis.load_data(df = self.df, emb=self.emb)
        self.error_analysis.process()
        # self.error_analysis.show_log()
    
    def output_result(self, ):
        # output metrics
        if self.metrics is None or 'all' not in self.metrics:
            self.logger.warning('all  not in metrics!')
            return
        values = self.metrics['all']
        self.indicator_panel['P0_Precision'] = values['precision_bev'] if 'precision_bev' in values else None
        self.indicator_panel['P0_Recall'] = values['recall_bev'] if 'recall_bev' in values else None
        self.indicator_panel['P0_F1_Score'] = values['f1_score_bev'] if 'f1_score_bev' in values else None
        self.indicator_panel['position_error'] = values['position_error'] if 'position_error' in values else None
        self.indicator_panel['yaw_error'] = values['yaw_error'] if 'yaw_error' in values else None
        if self.multi_result is not None:
            self.indicator_panel.update(self.multi_result)
        write_json(self.indicator_path, self.indicator_panel)




if __name__ == '__main__':
    temp_stat = StatsCam3D(config=StatsConfig)
    temp_stat.run()
