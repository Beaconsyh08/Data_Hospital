#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train_data_manager.py
@Time    :   2021/11/15 19:01:52
@Email   :   songyuhao@haomo.ai
@Author  :   YUHAO SONG 

Copyright (c) HAOMO.AI, Inc. and its affiliates. All Rights Reserved
"""


import pandas as pd
from src.data_manager.data_manager import DataManager


class TrainLidarManager(DataManager):
    """
    Summary
    -------
        This class is a sub-class of DataManager, but paticularly designed for Train data

    """

    def __init__(self, df: pd.DataFrame = None, df_path: str = None, cfg: dict = None) -> None:
        super(TrainLidarManager, self).__init__(df, df_path, cfg)


    def json_extractor(self, json_info: str, json_path: str) -> list:
        """
        Summary
        -------
            This function extract the json object and iterate the json object of train data

        Parameters
        ----------
            json_info: str
                pass in the json object to be extracted
            json_path: str
                the path of json object

        Returns
        -------
            list: list of dict objects to be put in dataframe

        """
        objs = json_info["objects"] 
        point_cloud_url = json_info["point_cloud_Url"]
        lidar_orientation = json_info["lidar_orientation"]
        lon, lat = super().lon_lat_extractor(json_info)
        city = super().location_decider(lon, lat)

        total_lst = []
        for obj in objs:
            info = dict()
            info["class_name"], info["id"], info["uuid"], info["point_count"], info["is_empty"] = obj["className"], obj["id"], obj["uuid"], obj["pointCount"], obj["isEmpty"]
            info["flag"], info["json_path"] = "train", json_path
            info["lidar_orientation"] = lidar_orientation
            info["lon"], info["lat"], info["city"] = lon, lat, city
            super().attributes_extractor_3d(obj, info)

            info["index_list"] = "/%s@%d" % (point_cloud_url, info["id"])
            total_lst.append(info)
        
        return total_lst
