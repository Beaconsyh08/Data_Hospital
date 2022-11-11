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
from configs.config import DataHospitalConfig
from src.data_manager.data_manager import DataManager

class TrainCam3dManager(DataManager):
    """
    Summary
    -------
        This class is a sub-class of DataManager, but paticularly designed for Train data

    """

    def __init__(self, df: pd.DataFrame = None, df_path: str = None, cfg: dict = None) -> None:
        super(TrainCam3dManager, self).__init__(df, df_path, cfg)

                
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
        img_url = json_info["imgUrl"]
        camera_orientation = json_info["camera_orientation"]
        camera_type = json_info["camera_type"]
        producer = json_info["producer"]
        timestamp = json_info["timestamp"] if type(json_info["timestamp"]) == type(1) else int(json_path.split("/")[-1][:-5])
        card_id = json_path.split("/")[-3]
        # print(json_info)
        img_width, img_height = super().width_height_extractor(json_info, json_info)
        lon, lat = super().lon_lat_extractor(json_info)
        city = super().location_decider(lon, lat)
        try:
            split_image_id = json_info["image_id"].split(".")
            car_id = split_image_id[1] if len(split_image_id) > 1 else img_url.split("/")[-3]
        except:
            car_id = img_url.split("/")[-3]
        
        total_lst = []
        
        for obj in objs:
            info = dict()
            # bbox =1, xy out; =2, xywh <0; =3 x+w/y+h out; =4 null
            # 2d_null =1, null; =2 []
            # 3d_null =1, null
            # coor =1, coor trans error
            # res =1, res error
            for error in DataHospitalConfig.TOTAL_ERROR_LIST + ["2d_null_error", "3d_null_error"]:
                info[error] = 0
            
            info["class_name"] = obj["className"]
            info["img_url"] = img_url
            info["card_id"], info["car_id"] = card_id, car_id
            info["img_width"], info["img_height"] = img_width, img_height
            info["uuid"] = obj.get("uuid")
            info["id"] = obj["id"]
            info["camera_orientation"], info["camera_type"], info["producer"] = camera_orientation, camera_type, producer
            info["lon"], info["lat"], info["city"] = lon, lat, city
            
            try:
                super().bbox_calculator(obj, info)    
            except TypeError:
                info["bbox_error"] = 3
                # continue
            
            try:
                super().attributes_extractor_2d(obj["2D_attributes"], info)
                has_2D = True
            except KeyError:
                has_2D = False
                info["2d_null_error"] = 1
            except TypeError:
                has_2D = False
                info["2d_null_error"] = 2
                
            if not has_2D:
                info["truncation"] = super().truncation_generator(info, img_width, img_height)    
            
            if obj.get("3D_attributes"):
                super().attributes_extractor_3d(obj["3D_attributes"], info)
                
                if has_2D:
                    super().define_priority(info)

            else:
                info["3d_null_error"] = 1
            
            super().date_time_extractor(timestamp, info)
            info["flag"], info["json_path"] = "train", json_path
            info["objects_amount"] = len(objs)

            info["index_list"] = "/%s@%d" % (img_url, info["id"])
            info["carday_id"] = car_id + "_" + str(info["date"])
                    
            total_lst.append(info)
            
        return total_lst
    