#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   qa_data_manager.py
@Time    :   2021/11/15 18:52:40
@Email   :   songyuhao@haomo.ai
@Author  :   YUHAO SONG 

Copyright (c) HAOMO.AI, Inc. and its affiliates. All Rights Reserved
"""


import json
import math
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from src.data_manager.data_manager import DataManager
import math


class QaCam3dManager(DataManager):
    """
    Summary
    -------
        This class is a sub-class of DataManager, but paticularly designed for QA data

    """

    def __init__(self, df: pd.DataFrame = None, df_path: str = None, cfg: dict = None) -> None:
        super(QaCam3dManager, self).__init__(df, df_path, cfg)
        
    
    def obj_iterator(self, objs_info: dict, json_path: str):
        """
        Summary
        -------
            This function iterate the json object of QA dataset.

        Parameters
        ----------
            objs_info: dict
                this info dict contains all required info for obj_iterator
            json_path: str
                the path of json object
                
        Returns
        -------
            list: list of dict objects to be put in dataframe
        
        """
        
        # "0": "2d good case"
        # "0-0": "2d & 3d correct"
        # "0-1": "3d incorrect"
        # "0-None": "p1, probably"
        # "1": "2d miss"
        # "2": "2d false"
        # "21": "2d iou < 0.5"
        # "22": "2d classification error"
        
        total_lst = []
        obj_lst = objs_info["objs"]
        
        for obj in obj_lst:
            # case_flag, bev_case_flag, iou, gt_obj, dt_obj = obj["result"], obj["result_bev"], obj["iou"], obj["gt_info"], obj["dt_info"]
            case_flag, iou, gt_obj, dt_obj = obj["result"], obj["iou"], obj["gt_info"], obj["dt_info"]
            bev_case_flag = obj.get("result_bev")
            case_flag, bev_case_flag = str(case_flag), str(bev_case_flag)
            
            # case_flag_lst = ["00", "01", "0None", "1", "2", "21None", "22None"]
            if case_flag in ["0", "21", "22"]:
                # DT INFO
                dt_info = self.general_obj_constructor(dt_obj, iou, "dt", objs_info, json_path, case_flag, bev_case_flag)
                dt_info["score"] = dt_obj["conf_score"]
                
                # GT INFO
                gt_info = self.general_obj_constructor(gt_obj, iou, "gt", objs_info, json_path, case_flag, bev_case_flag)
                gt_info["score"] = dt_obj["conf_score"]
                
                # DT & GT
                if gt_obj["3D_attributes"]:
                    try:
                        gt_info["points_count"] = gt_obj["3D_attributes"]["pointsCount"]
                    except KeyError:
                        gt_info["points_count"] = None
                    self.distance_attribute_calculator(dt_info, gt_info)
                    self.yaw_diff_calculator(dt_info, gt_info)
                
                if case_flag == "0" and bev_case_flag in ["0", "None"]:
                    gt_info["flag"] = "good"
                    
                    total_lst.append(gt_info)
                
                elif gt_obj["3D_attributes"] and bev_case_flag == "1" and gt_info["euclidean_dis_diff"] < 0.3:
                    gt_info["flag"] = "good"
                    
                    total_lst.append(gt_info)
                        
                else:
                    gt_info["flag"] = "miss"
                    dt_info["flag"] = "false"
                    
                    gt_info["peer_id"] = dt_info["index_list"]
                    dt_info["peer_id"] = gt_info["index_list"]

                    total_lst.append(gt_info)
                    total_lst.append(dt_info)

            # Miss Case, Only GT
            elif case_flag == "1":
                gt_info = self.general_obj_constructor(gt_obj, iou, "gt", objs_info, json_path, case_flag, bev_case_flag)
                if gt_obj["3D_attributes"]:
                    try:
                        gt_info["points_count"] = gt_obj["3D_attributes"]["pointsCount"]
                    except KeyError:
                        gt_info["points_count"] = None
                    gt_info["self_dis"] = np.linalg.norm(np.array([gt_info["x"], gt_info["y"]]))
                
                gt_info["flag"] = "miss"
                total_lst.append(gt_info)
                
            # False Case, Only DT
            elif case_flag == "2":
                dt_info = self.general_obj_constructor(dt_obj, iou, "dt", objs_info, json_path, case_flag, bev_case_flag)
                dt_info["score"] = dt_obj["conf_score"]
                dt_info["self_dis"] = np.linalg.norm(np.array([dt_info["x"], dt_info["y"]]))
                
                dt_info["flag"] = "false"
                total_lst.append(dt_info)

        return total_lst
    
    
    def json_extractor(self, json_info: json, json_path: str) -> list:
        """
        Summary
        -------
            This function extract the json object and pass the wanted objects to the qa_obj_iterator

        Parameters
        ----------
            json_info: json
                pass in the json object to be extracted
            json_path: str
                the path of json object

        Returns
        -------
            list: list of dict objects for ground truth and detection
            
        """

        # get the required information from the json, ground truth, detection and badcases
        required_info = self.get_basic_infos(json_info, json_path)
        # print(required_info)

        # if not os.path.isfile("/" + required_info["img_url"]):
        #     print("???")
        #     return []

        # iterate the cases
        return self.obj_iterator(required_info, json_path)
    
    
    def get_basic_infos(self, json_info: json, json_path: str) -> dict:
        """
        Summary
        -------
            This function save the basic infos into a dictionary, which extracted from orginal json

        Parameters
        ----------
            json_info: json
                pass in the json object to be extracted

        Returns
        -------
            dict: a dict contains all required infos for further processing
            
        """
        res = dict()
        res["camera_orientation"] = json_info["camera_orientation"]
        # assert json_info["camera_orientation"] == "front_middle_camera", print(json_info["camera_orientation"])
        res["camera_type"] = json_info["camera_type"]
        res["img_url"] = json_info["imgUrl"]
        res["producer"] = json_info["producer"]
        # Convert the Unix Time to Local Date Time, ignore microsecond
        # date_time = datetime.fromtimestamp(int(str()[:10]))
        # res["date"] = date_time.date()
        # res["time"] = date_time.time()
        super().width_height_extractor(res, json_info)
        try:
            timestamp = json_info["timestamp"] if type(json_info["timestamp"]) == int else int(json_path.split("/")[-1][:-5])
        except ValueError:
            timestamp = int(json_path.split("/")[-1][-21:-5])

        super().date_time_extractor(timestamp, res)
        res["objs"] = json_info["cases"]
        
        return res
    
    
    def input_obj_basic_infos(self, obj: dict, res_obj: dict, iou: float, dtgt: str, objs_info: dict, json_path: str, case_flag: str, bev_case_flag: str) -> None:
        """
        Summary
        -------
            This function dump all basic infos to the result object dict
            
        Parameters
        ----------
            obj: dict
                the dt or gt object dict
            res_obj: dict
                the result object dict to be dumped infos
            iou: float
            dtgt: str
                dt: detection; gt: ground truth
            objs_info: dict
                the dict contains main infos
            json_path: str
            case_flag: str
            bev_case_flag: str

        """
        
        TYPE_MAP = {'car': 'car', 'van': 'car', 
            'truck': 'truck', 'forklift': 'truck',
            'bus':'bus', 
            'rider':'rider',
            'rider-bicycle': 'rider', 'rider-motorcycle':'rider', 
            'rider_bicycle': 'rider', 'rider_motorcycle':'rider',
            'bicycle': 'bicycle', 'motorcycle': 'bicycle',
            'tricycle': 'tricycle', 'closed-tricycle':'tricycle', 'open-tricycle': 'tricycle', 
            'closed_tricycle':'tricycle', 'open_tricycle': 'tricycle', 'pedestrian': 'pedestrian',
            'static': 'static', 'trafficCone': 'static', 'water-filledBarrier': 'static', 'other': 'static', 'accident': 'static', 'construction': 'static', 'traffic-cone': 'static', 'other-vehicle': 'static', 'attached': 'static', 'accident': 'static', 'traffic_cone': 'static', 'other-static': 'static', 'water-filled-barrier': 'static', 'other_static': 'static', 'water_filled_barrier': 'static', 'dynamic': 'static', 'other_vehicle': 'static', 'trafficcone': 'static', 'water-filledbarrier': 'static',
            }
        
        
        # 'vehicle': ['car', 'bus', 'truck', 'van', 'forklift'],
        # 'vru': ['pedestrian', 'rider', 'bicycle', 'tricycle', 'rider-bicycle', 'rider-motorcycle', 'rider_bicycle', 'rider_motorcycle', 'closed-tricycle', 'open-tricycle', 'closed_tricycle', 'open_tricycle'],
        # 'static': ['trafficCone', 'water-filledBarrier', 'other', 'accident', 'construction']
        
        # ALL_CATEGORY = ['car', 'bus', 'truck', 'van', 'forklift', 'pedestrian', 'rider', 'bicycle', 'tricycle', 'rider-bicycle', 'rider-motorcycle', 'rider_bicycle', 'rider_motorcycle', 'closed-tricycle', 'open-tricycle', 'closed_tricycle', 'open_tricycle']
        
        
        res_obj["id"] = int(obj["id"])
        # try:
        res_obj["class_name"] = TYPE_MAP[obj['className']]
        # except KeyError:
        #     res_obj["class_name"] = static

        res_obj["iou"] = iou
        res_obj["dtgt"] = dtgt
        res_obj["index_list"] = "/%s@%d@%s" % (objs_info["img_url"], res_obj["id"], res_obj["dtgt"])
        res_obj["json_path"] = json_path
        res_obj["camera_orientation"] = objs_info["camera_orientation"]
        res_obj["camera_type"] = objs_info["camera_type"]
        res_obj["producer"] = objs_info["producer"]
        res_obj["date"] = objs_info["date"]
        res_obj["time"] = objs_info["time"]
        res_obj["case_flag"] = case_flag
        res_obj['bev_case_flag'] = bev_case_flag
        res_obj["img_width"] = objs_info["img_width"]
        res_obj["img_height"] = objs_info["img_height"]
    
    
    def distance_attribute_calculator(self, dt_info: dict, gt_info: dict) -> None:
        """
        Summary
        -------
            Calculate the features about distance
            
        Parameters
        ----------
            dt_info: dict
                the detectino info
            gt_info: dict
                the ground truth info    
                
        """
        
        gt_info["euclidean_dis_diff"] = np.linalg.norm(np.array([dt_info["x"], dt_info["y"]]) - np.array([gt_info["x"], gt_info["y"]]))
        dt_info["euclidean_dis_diff"] = gt_info["euclidean_dis_diff"]
        gt_info["self_dis"] = np.linalg.norm(np.array([gt_info["x"], gt_info["y"]]))
        dt_info["self_dis"] = np.linalg.norm(np.array([dt_info["x"], dt_info["y"]]))
        gt_info["dis_ratio"] = gt_info["euclidean_dis_diff"] / gt_info["self_dis"]
        dt_info["dis_ratio"] = dt_info["euclidean_dis_diff"] / dt_info["self_dis"]
        
    
    def yaw_diff_calculator(self, dt_info: dict, gt_info: dict) -> None:
        """
        Summary
        -------
            Calculate the difference of yaw value
            
        Parameters
        ----------
            dt_info: dict
                the detectino info
            gt_info: dict
                the ground truth info          
        
        """
        
        gt_info["yaw_diff"] = abs(gt_info["yaw"] - dt_info["yaw"]) if abs(gt_info["yaw"] - dt_info["yaw"]) <= math.pi else 2 * math.pi - abs(gt_info["yaw"] - dt_info["yaw"])
        dt_info["yaw_diff"] = gt_info["yaw_diff"]
        
    
    def general_obj_constructor(self, obj: dict, iou: float, dtgt: str, objs_info: dict, json_path:str, case_flag: str, bev_case_flag: str) -> None:
        """
        Summary
        -------
            Connt how many objects located in the single images
            
        Parameters
        ----------
            obj: dict
                the dt or gt object dict
            iou: float
            dtgt: str
                dt: detection; gt: ground truth
            objs_info: dict
                the dict contains main infos
            json_path: str
            case_flag: str
            bev_case_flag: str
        
        """
        
        info = dict()
        self.input_obj_basic_infos(obj, info, iou, dtgt, objs_info, json_path, case_flag, bev_case_flag)
        super().bbox_calculator(obj, info)
        super().bbox_checker(info) 
        try:
            super().attributes_extractor_2d(obj["2D_attributes"], info)
        except KeyError:
            pass
        if obj["3D_attributes"]:
            super().attributes_extractor_3d(obj["3D_attributes"], info)
            self.volume_calcultor(info)
            self.rel_position_calculator(info)
            super().define_priority(info)
            
        self.objects_amount_counter(objs_info, info)
        
        return info
    
    
    def volume_calcultor(self, res_obj: dict) -> None:
        """
        Summary
        -------
            Calculte the volume of object
            
        Parameters
        ----------
            res_obj:
                dump the result to the res_obj dict
        
        """
        
        res_obj["volume"] = res_obj["width"] * res_obj["height"] * res_obj["length"]
        
    
    def objects_amount_counter(self, objs_info: dict, res_obj: dict) -> None:
        """
        Summary
        -------
            Connt how many objects located in the single images
            
        Parameters
        ----------
            objs_info:
                the dict contains the main infos
            res_obj:
                dump the result to the res_obj dict
        
        """
        
        res_obj["objects_amount"] = len(objs_info["objs"])
        
        
    def rel_position_calculator(self, res_obj: dict) -> None:
        """
        Summary
        -------
            Calculate the relative position of the object in terms of self-car
            
        Parameters
        ----------
            res_obj:
                the object contains the required element, and the place dump the calculation result
        
        """
        
        x = res_obj["x"]
        y = res_obj["y"]

        rad = math.atan(abs(y) / abs(x)) if abs(x) != 0 else 0 
        rad_final = 0
        if x > 0 and y > 0:
            rad_final = - rad
        elif x < 0 and y > 0:
            rad_final = - (math.pi - rad)
        elif x > 0 and y < 0:
            rad_final = rad
        elif x <0 and y < 0:
            rad_final = math.pi - rad
            
        res_obj["rel_position"] = rad_final
