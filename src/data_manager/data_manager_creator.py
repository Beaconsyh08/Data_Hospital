#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_manager_creator.py
@Time    :   2021/11/15 18:50:27
@Email   :   songyuhao@haomo.ai
@Author  :   YUHAO SONG 

Copyright (c) HAOMO.AI, Inc. and its affiliates. All Rights Reserved
"""


from src.data_manager.data_manager import DataManager
from src.data_manager.qa_cam2d_manager import QaCam2dManager
from src.data_manager.qa_cam3d_manager import QaCam3dManager
from src.data_manager.qa_lidar_manager import QaLidarManager
from src.data_manager.qa_simple_manager import QaSimpleManager
from src.data_manager.ret_cam2d_manager import RetrievalCam2dManager
from src.data_manager.ret_cam3d_manager import RetrievalCam3dManager
from src.data_manager.ret_lidar_manager import RetrievalLidarManager
from src.data_manager.train_cam2d_manager import TrainCam2dManager
from src.data_manager.train_cam3d_manager import TrainCam3dManager
from src.data_manager.train_lidar_manager import TrainLidarManager
from src.data_manager.inf_cam2d_cloud_manager import InfCam2dCloudDataManager
from src.data_manager.inf_cam3d_cloud_manager import InfCam3dCloudDataManager
from src.data_manager.inf_lidar_cloud_manager import InfLidarCloudDataManager
from src.data_manager.inf_cam2d_car_manager import InfCam2dCarDataManager
from src.data_manager.inf_cam3d_car_manager import InfCam3dCarDataManager
from src.data_manager.inf_lidar_car_manager import InfLidarCarDataManager

from src.utils.logger import get_logger

logger = get_logger()

def data_manager_creator(cfg: dict) -> DataManager:
    """
    Summary
    -------
        create data manger instance based on the config dict of different type
    
    Parameters
    ----------
        cfg: dict

    Returns
        DataManager: subclass of DataManager
    -------
    
    """
    
    if cfg.DATA_TYPE.lower() == "qa_cam2d":
        manager = QaCam2dManager
    elif cfg.DATA_TYPE.lower() == "qa_cam3d":
        manager = QaCam3dManager
    elif cfg.DATA_TYPE.lower() == "qa_lidar":
        manager = QaLidarManager
    elif cfg.DATA_TYPE.lower() == "qa_simple":
        manager = QaSimpleManager
    elif cfg.DATA_TYPE.lower() == "train_cam2d":
        manager = TrainCam2dManager
    elif cfg.DATA_TYPE.lower() == "train_cam3d":
        manager = TrainCam3dManager
    elif cfg.DATA_TYPE.lower() == "train_lidar":
        manager = TrainLidarManager
    elif cfg.DATA_TYPE.lower() == "ret_cam2d":
        manager = RetrievalCam2dManager
    elif cfg.DATA_TYPE.lower() == "ret_cam3d":
        manager = RetrievalCam3dManager
    elif cfg.DATA_TYPE.lower() == "ret_lidar":
        manager = RetrievalLidarManager
    elif cfg.DATA_TYPE.lower() == "inf_cam2d_cloud":
        manager = InfCam2dCloudDataManager
    elif cfg.DATA_TYPE.lower() == "inf_cam3d_cloud":
        manager = InfCam3dCloudDataManager
    elif cfg.DATA_TYPE.lower() == "inf_lidar_cloud":
        manager = InfLidarCloudDataManager
    elif cfg.DATA_TYPE.lower() == "inf_cam2d_car":
        manager = InfCam2dCarDataManager
    elif cfg.DATA_TYPE.lower() == "inf_cam3d_car":
        manager = InfCam3dCarDataManager
    elif cfg.DATA_TYPE.lower() == "inf_lidar_car":
        manager = InfLidarCarDataManager
    if cfg.DATA_TYPE is None:
        logger.warning("No predifined data manager type exist.")
        manager = DataManager 
    logger.info("Data Manager Instance Created")
    return manager(cfg=cfg)
