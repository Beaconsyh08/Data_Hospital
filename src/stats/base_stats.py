#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   base_stats.py
@Time    :   2022/01/03 16:27:54
@Author  :   chenminghua 
'''

from genericpath import exists
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm
from configs.config import Config, StatsConfig, VisualizationConfig,EvalDataFrameConfig
from src.data_manager.data_manager_creator import data_manager_creator
from src.utils.logger import get_logger
import src.data_manager.data_manager as dm
from src.visualization.visualization import Visualizer


class BaseStats(object):
    def __init__(self, config: dict = None, df: pd.DataFrame = None,
                       df_path: str = None, emb_path: str = None) -> None:
        super().__init__()
        self.config = config
        self.df = df
        self.df_path = df_path
        self.df_multi_version = []
        self.emb_path = emb_path
        self.logger = get_logger()
        self.visualizer = Visualizer(VisualizationConfig)
    
    def run(self, ):
        self._load_data()

    def _load_data(self, ):
        if self.config is not None:
            self.logger.info("Loading Data From StatsConfig")
            self.df_path = self.config.DATAFRAME_PATH
            self.df = dm.load_from_pickle(self.config.DATAFRAME_PATH)
            self.emb = np.load(self.config.EMB_PATH) if os.path.exists(self.config.EMB_PATH) else None
            for path in self.config.MULTI_VERSIONS:
                if not os.path.exists(path):
                    continue
                EvalDataFrameConfig.JSON_PATH = path
                data_manager = data_manager_creator(EvalDataFrameConfig)
                data_manager.load_from_json()
                # retrieval_dataframe = retrieval_data.getter()
                # df = dm.load_from_json(path)
                self.df_multi_version.append(data_manager.getter())
        else:
            if self.df_path is not None:
                self.df = dm.load_from_pickle(self.df_path)
            if self.emb_path is not None:
                self.emb = np.load(self.emb_path)
        
        self.dm = dm.DataManager(df=self.df)
        