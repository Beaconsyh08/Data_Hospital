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
import math
from PIL import Image
from configs.config import LogisticDoctorConfig

# WEY MOKA
self_width = 1.96
self_length = 4.875

class TrainCam3dManager(DataManager):
    """
    Summary
    -------
        This class is a sub-class of DataManager, but paticularly designed for Train data

    """

    def __init__(self, df: pd.DataFrame = None, df_path: str = None, cfg: dict = None) -> None:
        super(TrainCam3dManager, self).__init__(df, df_path, cfg)
    

    def coor_checker_car(self, info: dict) -> None:
        """
        Summary
        -------
            Check if the coordinate system trans has error, and assign the corresponding flag for the coor_error
                1: trans error
                
        Parameters
        ----------
            info: dict
                the info json object
                
        """
        
        ori = info["camera_orientation"]
        x, y, z, h = info["x"], info["y"], info["z"], info["height"]
        
        if info["truncation"] == 0 or info["truncation"] == None:
            x = x - 2 * self_length / 3

            # tolerance for side cam
            side_co_x = math.tan(10 * math.pi/180)
            side_cox = abs(x) * side_co_x
            
            side_co_y = math.tan(10 * math.pi/180)

            # tolerance for front & rear cam
            front_co = math.tan(60* math.pi/180)

            if ori == "front_left_camera":
                y = y - self_width / 2
                side_coy = abs(y) * side_co_y
                if x < -side_coy or y < -side_cox:
                    info["coor_error"] = 1
                    
            elif ori == "front_right_camera":
                y = y + self_width / 2
                side_coy = abs(y) * side_co_y
                if x < -side_coy or y > side_cox:
                    info["coor_error"] = 1
                    
            elif ori == "rear_left_camera": 
                y = y - self_width / 2
                side_coy = abs(y) * side_co_y
                if x > side_coy or y < -side_cox:
                    info["coor_error"] = 1
                    
            elif ori == "rear_right_camera":
                y = y + self_width / 2
                side_coy = abs(y) * side_co_y
                if x > side_coy or y > side_cox:
                    info["coor_error"] = 1
                    
            elif ori == "front_middle_camera": 
                front_side_cox = abs(x) * front_co
                if x < 0 or y < - front_side_cox or y > front_side_cox:
                    info["coor_error"] = 1
                    
            elif ori == "rear_middle_camera": 
                x = x + 2 * self_length / 3
                front_side_cox = abs(x) * front_co
                if x > 0 or y < - front_side_cox or y > front_side_cox: 
                    info["coor_error"] = 1
            
        else:
            if z - h / 2 < -1:
                if abs(x) < 10 and abs(y) < 10:
                    info["coor_error"] = 3
                else:
                    info["coor_error"] = 2
                    
    
    def coor_checker_lidar(self, info: dict) -> None:
        """
        Summary
        -------
            Check if the coordinate system trans has error, and assign the corresponding flag for the coor_error
                1: trans error
                
        Parameters
        ----------
            info: dict
                the info json object
                
        """
        
        ori = info["camera_orientation"]
        x, y, z, h = info["x"], info["y"], info["z"], info["height"]
        
        if info["truncation"] == 0 or info["truncation"] == None:
            y = y + self_length / 6

            # tolerance for side cam
            side_co_y = math.tan(10 * math.pi/180)
            side_coy = abs(y) * side_co_y
            side_co_x =  math.tan(10 * math.pi/180)

            # tolerance for front & rear cam
            front_co = math.tan(60 * math.pi/180)

            if ori == "front_left_camera":
                x = x - self_width / 2
                side_cox = abs(x) * side_co_x
                if x < -side_coy or y > side_cox:
                    info["coor_error"] = 1
                    
            if ori == "front_right_camera":
                x = x + self_width / 2
                side_cox = abs(x) * side_co_x
                if x > side_coy or y > side_cox:
                    info["coor_error"] = 1
                    
            if ori == "rear_left_camera":
                x = x - self_width / 2
                side_cox = abs(x) * side_co_x
                if x < -side_coy or y < -side_cox:
                    info["coor_error"] = 1
                    
            if ori == "rear_right_camera":
                x = x + self_width / 2 
                side_cox = abs(x) * side_co_x
                if x > side_coy or y < -side_cox:
                    info["coor_error"] = 1
                    
            if ori == "front_middle_camera":
                front_side_coy = abs(y) * front_co
                if y > 0 or x < - front_side_coy or x > front_side_coy:
                    info["coor_error"] = 1
                    
            if ori == "rear_middle_camera":
                y = y - self_length / 6 - self_length / 2
                front_side_coy = abs(y) * front_co
                if y < 0 or x < - front_side_coy or x > front_side_coy:
                    info["coor_error"] = 1
        
        else:
            if z - h / 2 > -1:
                if abs(x) < 10 and abs(y) < 10:
                    info["coor_error"] = 3
                else:
                    info["coor_error"] = 2
                

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
            info["class_name"] = obj["className"]
            info["card_id"], info["car_id"] = card_id, car_id
            
            try:
                info["uuid"] = obj["uuid"]
            except:
                pass
            
            info["id"] = obj["id"]
            info["camera_orientation"], info["camera_type"], info["producer"] = camera_orientation, camera_type, producer
            info["lon"], info["lat"], info["city"] = lon, lat, city
            
            # bbox =1, xy out; =2, xywh <0; =3 x+w/y+h out; =4 null
            # 2d_null =1, null; =2 []
            # 3d_null =1, null
            # coor =1, coor trans error
            info["bbox_error"], info["2d_null_error"], info["3d_null_error"], info["coor_error"] = 0, 0, 0, 0
            
            try:
                super().bbox_calculator(obj, info)    
                super().bbox_checker(info, img_width, img_height) 
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
                
            if obj["3D_attributes"]:
                super().attributes_extractor_3d(obj["3D_attributes"], info)
                try:
                    if LogisticDoctorConfig.COOR == "Car":
                        self.coor_checker_car(info)
                    else:
                        self.coor_checker_lidar(info)
                except KeyError:
                    pass
                
                    if has_2D:
                        super().define_priority(info)

            else:
                info["3d_null_error"] = 1
            
            super().date_time_extractor(timestamp, info)
            info["flag"], info["json_path"] = "train", json_path
            info["objects_amount"] = len(objs)

            info["index_list"] = "/%s@%d" % (img_url, info["id"])
            
            info["has_error"] = 1 if info["bbox_error"] != 0 or info["coor_error"] != 0 else 0
            info["carday_id"] = car_id + "_" + str(info["date"])
                    
            total_lst.append(info)
            
        return total_lst
    