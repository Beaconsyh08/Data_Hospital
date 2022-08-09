#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   analyze_qa.py
@Time    :   2021/10/28 15:21:26
@Author  :   chenminghua 
'''
import os
from pathlib import Path

import src.data_manager.data_manager as dm
from configs.config import (ClusterConfig, Config, EmbeddingConfig,
                            EvalDataFrameConfig, OutputConfig, SamplerConfig,
                            StatsConfig, TrainDataFrameConfig)
from src.classification.clustering import Clustering
from src.data_manager.data_manager import load_from_pickle
from src.data_manager.data_manager_creator import data_manager_creator
from src.data_manager.pre_process.tagging import Tagging
from src.data_manager.qa_cam3d_manager import QaCam3dManager
from src.feature_embedding.compute_embedding import ComputeEmbedding
from src.sampler.creat_sampler import create_sampler
from src.stats.create_stats import StatsCreator
# from src.stats.single_stats import SingleStats
from src.utils.file_io import save_query_files
from src.utils.logger import get_logger
from tqdm import tqdm


class QAAnalysis(object):
    """
    Description: Determine the query to retrieve by analyzing the training 
                 and test data sets
    """
    def __init__(self, ) -> None:
        super().__init__()
        # set logger
        self.logger = get_logger()
        # set path
        self.input_eval_path = Path(EvalDataFrameConfig.JSON_PATH)
        self.input_train_path = Path(TrainDataFrameConfig.JSON_PATH)
        self.output_query_dir = Path(OutputConfig.QUERY_OUTPUT_DIR)
        self.output_others_dir = Path(OutputConfig.OTHERS_OUTPUT_DIR)
        # init modules
        self.qa_data_engine = data_manager_creator(EvalDataFrameConfig)
        self.train_data_engine = data_manager_creator(TrainDataFrameConfig)
        # self.retrieval_data_engine = data_manager_creator(RetrievalDataFrameConfig)
        self.tagging = Tagging()
        self.emb_computer = ComputeEmbedding(EmbeddingConfig)
        self.stats = StatsCreator()


    def process(self, ):
        """
        Description: Dataflow: load data -> compute embedding -> clustering -> sampling
        Param: 
        Returns: 
        """
        # load dataset
        self.load_data()
        
        # pre_process data
        # self.tagging.process()

        # compute embedding
        # self.emb_computer.process()
        
        # clustering
        # self.cluster = Clustering()
        # emb_id = [_ for _ in range(len(dm.load_from_pickle(ClusterConfig.DATAFRAME_PATH_P0)))]
        # self.cluster.clustering(emb_id)
        
        # stats and vis
        # self.stats = SingleStats(cfg=StatsConfig)
        self.stats.run()
        # self.stats.plot_all()

        # sampling
        # self.sampler = create_sampler(SamplerConfig)
        # self.sampler.sample()

        # # output analysis results
        # df_sampler = load_from_pickle(SamplerConfig.DATAFRAME_PATH)
        # save_query_files(self.output_query_dir, df_sampler)
        # self.sampler.save_to_pic(OutputConfig.DATAFRAME_OUTPUT_PATH)


    def load_data(self, ):
        # load data
        if not os.path.exists(EvalDataFrameConfig.JSON_PATH):
            self.logger.error("%s not exist", EvalDataFrameConfig.JSON_PATH)
            return
        
        # load QA evaluation results
        self.qa_data_engine.load_from_json()
        self.qa_data_engine.save_to_pickle(EvalDataFrameConfig.EVAL_DATAFRAME_PATH)
        
        
        # load train data
        if os.path.exists(TrainDataFrameConfig.JSON_PATH):
            self.train_data_engine.load_from_json()
            self.train_data_engine.save_to_pickle(TrainDataFrameConfig.TRAIN_DATAFRAME_PATH)
            if len(self.train_data_engine.getter()) > len(self.qa_data_engine.getter()):
                self.train_data_engine.sampler(len(self.qa_data_engine.getter()))
                # self.train_data_engine.sampler(0)   
        
        
        # combine train_dataframe and qa_dataframe
        if os.path.exists(TrainDataFrameConfig.JSON_PATH) :
            self.combined_data_engine = dm.merge_dataframe_rows(self.qa_data_engine.getter(), self.train_data_engine.getter())
            self.combined_data_engine.save_to_pickle(Config.DATAFRAME_PATH)
        else:
            self.qa_data_engine.save_to_pickle(Config.DATAFRAME_PATH)
        
        QaCam3dManager(df=self.qa_data_engine.df[(self.qa_data_engine.df.priority == "P0") & (self.qa_data_engine.df.flag != "good") & (self.qa_data_engine.df.class_name.isin(StatsConfig.ALL_CATEGORY))]).save_to_pickle(EvalDataFrameConfig.DATAFRAME_PATH_P0)
            

if __name__ == '__main__':
    qa_analysis = QAAnalysis()
    qa_analysis.process()
    