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
from src.data_manager.qa_cam3d_manager import QaCam3dManager
from src.data_manager.train_cam3d_manager import TrainCam3dManager
from src.data_manager.train_lidar_manager import TrainLidarManager

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
    
    if cfg.DATA_TYPE.lower() == "qa_cam3d":
        manager = QaCam3dManager
    elif cfg.DATA_TYPE.lower() == "train_cam3d":
        manager = TrainCam3dManager
    elif cfg.DATA_TYPE.lower() == "train_lidar":
        manager = TrainLidarManager
    if cfg.DATA_TYPE is None:
        logger.warning("No predifined data manager type exist.")
        manager = DataManager 
    logger.info("Data Manager Instance Created")
    return manager(cfg=cfg)
