#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   analysis_data_engine.py
@Time    :   2021/11/11 16:48:51
@Email   :   songyuhao@haomo.ai
@Author  :   YUHAO SONG 

Copyright (c) HAOMO.AI, Inc. and its affiliates. All Rights Reserved
"""


import json
import os
import pickle
import sys
from abc import abstractmethod
from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import List

import pandas as pd
from tqdm import tqdm
from PIL import Image
from shapely.geometry import Point, Polygon
from configs.config import Config
from src.utils.logger import get_logger
from itertools import zip_longest



class DataManager:
    """
    Summary
    -------
        An instance of this class could manage the data in a form of pandas.DataFrame from jsons
    """
    
    def __init__(self, df: pd.DataFrame = None, df_path: str = None, cfg: dict = None) -> None:
        """
        Summary
        -------
            The Constructor for the class, but how to build it
        
        Parameters
        ----------
            df (optional): pd.DataFrame, default = "None"
                the dataframe
            df_path (optional): str, default = "None"
                the dataframe pickle pat
            cfg (optional): dict, default = "None"
                the config dict
            
        """

        self.df = df if df_path is None else load_from_pickle(df_path)
        self.city_path = "/share/analysis/city_boundary/"
        self.city_polygons = self._city_polygon_constructor()
        self.logger = get_logger()
        self.cfg = cfg


    def getter(self) -> pd.DataFrame:
        """
        Summary
        -------
            Return the dataframe of DataManager instance, equal to instance.df
        
        Returns
        -------
            pd.DataFrame
            
        """
        
        return self.df
    

    def get_cols(self, cols: List[str]) -> pd.DataFrame:
        """
        Summary
        -------
            return wantted category info in dataframe

        Parameters
        ----------
            cols: List[str]
                list of string of wantted category

        Returns
        -------
            pd.DataFrame
        
        """
        
        return self.df.loc[:, cols]

    
    def get_col_values(self, col_name: str) -> list:
        """
        Summary
        -------
            Return the wanted one column in list fo value
        
        Parameters
        ----------
            col_name: str
                the name of the specific column

        Returns
        -------
            list: the list of wantted value
            
        """
        
        return self.df[col_name].to_list()
    

    def get_rows(self, rows: List[str]) -> pd.DataFrame:
        """
        Summary
        -------
            Return the wantted rows in dataframe

        Parameters
        ----------
            rows: List[str]
                list of string of wantted rows

        Returns
        -------
            pd.DataFrame
        
        """
        
        return self.df.loc[rows, :]


    def sampler(self, n: int) -> None:
        """
        Summary
        -------
            Sampling and resize the dataframe

        Parameters
        ----------
            n: int
                sampel to n rows
                
        """
        
        self.df = self.df.sample(n = n)


    @abstractmethod
    def json_extractor(self, json_info: json, json_path: str, count: int) -> List[dict]:
        """
        Summary
        -------
            Abstract method for the subclass using
            Extracto the json and return list of objects(instances)

        Parameters
        ----------
            json_info: json
                the json object
            json_path: str
                the json path in string

        Returns
        -------
            List[dict]: a list of instances, info in dict
        
        """

        pass
    
    
    def truncation_generator(self, info: dict, img_width: int, img_height: int) -> int:
        if (info["bbox_x"] + info["bbox_w"] > 0.995 * img_width or info["bbox_x"] <  0.005 * img_width) and Config.TYPE_MAP[info["class_name"]] not in ["pedestrian", "static"]:
            return 1
        else:
            return 0
        
    
    def define_priority(self, obj: dict) -> str:
        """
        Summary
        -------
            Define the priority of the case
            
        Parameters
        ----------
            obj: dict

        Returns
        -------
            str: the priority of the case, ["P0", "P1", "P2"]
                
        """
        
        try:
            occlusion, crowding = obj["occlusion"], obj["crowding"]
        except KeyError:
            occlusion, crowding = None, None
            
        pos_x, pos_y = obj["x"], obj["y"]
        priority = 'P0'
                
        if obj["camera_orientation"] == "front_middle_camera":
            if  pos_x > 60 or abs(pos_y) > 10 or pos_x < 0:
                priority = 'P1'
        else:
            if abs(pos_x) > 20 or abs(pos_y) > 10 or abs(pos_y) < 1:
                priority = 'P1'  
        if (occlusion != None and int(occlusion) > 0) or (crowding != None and int(crowding) > 0):
            priority = 'P1'
            
        obj["priority"] = priority
            

    def bbox_calculator(self, obj: dict, res_obj: dict) -> None:
        """
        Summary
        -------
            Extract the bbox info in json and Calculate the size of bbox, then put result in result object
        
        Parameters
        ----------
            obj: dict
                the json object of the object contains bbox info
            res_obj: dict
                the result object
            
        """
        
        x, y, w, h = [_ for _ in obj["bbox"]]
        b_size = w * h
        size = "small" if b_size <= 32 * 32 else ("medium" if 32 * 32 < b_size <= 96 * 96 else "large")
        res_obj["bbox_x"], res_obj["bbox_y"], res_obj["bbox_w"], res_obj["bbox_h"], res_obj["bbox_size"], res_obj["size"] = x, y, w, h, b_size, size


    def attributes_extractor_2d(self, attributes: dict, res_obj: dict) -> None:
        """
        Summary
        -------
            Extract the 2d attributes in the atrribute and put in the result object
            
        Parameters
        ----------
            attributes: dict
                the json of attributes
            res_obj: dict
                the result object
        
        """
        
        attr_list = ["occlusion", "truncation", "crowding", "direction", "onRoad", "pose", "crowded", "group"]
        for attr in attr_list:
            try:
                res_obj[attr] = int(attributes[attr])
            except ValueError:
                res_obj[attr] = attributes[attr]
            except KeyError:
                res_obj[attr] = None
            except TypeError:
                res_obj[attr] = None
            
    
    def attributes_extractor_3d(self, attributes: dict, res_obj: dict) -> None:        
        """
        Summary
        -------
            Extract the 3d attributes in the atrribute and put in the result object
            
        Parameters
        ----------
            attributes: dict
                the json of attributes
            res_obj: dict
                the result object
        
        """
        
        res_obj["height"], res_obj["length"], res_obj["width"] = attributes["dimension"].values()
        res_obj["x"], res_obj["y"], res_obj["z"] = attributes["position"].values()
        res_obj["pitch"], res_obj["roll"], res_obj["yaw"] = attributes["rotation"].values()
        
    
    def date_time_extractor(self, timestamp_input: str, res_obj: dict):
        date_time = datetime.fromtimestamp(int(str(timestamp_input)[:10]))
        res_obj["date"] = date_time.date()
        res_obj["time"] = date_time.time()
    
    
    def width_height_extractor(self, info: dict, json_info: dict) -> int:
        try:
            width, height = json_info["width"], json_info["height"]
        except:
            width, height = json_info["img_width"], json_info["img_height"]
            
        if type(width) != int or type(height) != int:
            try:
                image_szie = Image.open("/" + info["img_url"]).size
            except:
                image_szie = Image.open("/" + info["imgUrl"]).size
            width, height = image_szie[0], image_szie[1]

        if width < height:
            width, height = height, width
            
        info["img_width"], info["img_height"] = width, height
        
        return width, height
        
    
    def _city_polygon_constructor(self, ) -> dict:
        """
        Summary
        -------
            Construct Polygon for all avaliable cities
        
        Returns
        -------
            dict: dict contains all cities' name and Polygon
            
        """
        
        res_dict = dict()
        for city in os.listdir(self.city_path):
            json_path = os.path.join(self.city_path, city)
            with open(json_path) as json_obj:
                json_info = json.load(json_obj)
                coords = json_info["features"][0]["geometry"]["coordinates"][0]
                city = city.split('.')[0]
                if city.lower() == "tianjin":
                    coord = coords
                else:
                    assert len(coords) == 1, print("MultiPolygon! %s" % city)
                    coord = coords[0]
                    
                coord = [Point(_[0], _[1]) for _ in coord]
                poly = Polygon(coord)
                res_dict[city] = poly
                
        return res_dict


    def lon_lat_extractor(self, obj: dict):
        """
        Summary
        -------
            Extract longitude and latitude from json if it has rtk_data
        
        Returns
        -------
            Point: longitude and latitude in Point Format
            
        """
            
        try:
            lon_lat = obj["rtk_data"]["lon_lat_height"]
            return lon_lat["longitude"], lon_lat["latitude"]
        except KeyError:
            return None, None
        except TypeError:
            return None, None
            
    
    def location_decider(self, lon: float, lat: float) -> str:
        """
        Summary
        -------
            Decide if the point within city's polygon
        
        Returns
        -------
            str: the name of city
            
        """
        
        
        if lon is not None and lat is not None:
            p = Point(lon, lat)
            for city, polygon in self.city_polygons.items():
                if p.within(polygon):
                    return city
            return "UNK"
        else:
            return "NORTK"
        
        
        
    def bigcar_decider(self, obj: dict) -> str:
        """
        Summary
        -------
            Decide if the point within city's polygon
        
        Returns
        -------
            str: the name of city
            
        """
        
        bbox_w, bbox_h = obj["bbox_w"], obj["bbox_h"]
        
        bbox_w * bbox_h > 1920 * 1080 / 6
        return True
    
    
    def load_from_json(self, data_type: str = None, json_type: str = None, load_path: str = None) -> None:
        """
        Summary
        -------
            Loading from json
        
        Parameters
        ----------
            data_type (optional): str, default = "None"
                type of the data, chosen from ["qa", "train", "ret", "inf"]
            json_type (optional): str, default = "None"
                type of the json data, chosen from ["txt", "folder"]
            json_path (optional): str, default = "None"
                the path of json 
        """

        # set parameters from config dict
        data_type = self.cfg.DATA_TYPE if data_type is None else data_type
        json_type = self.cfg.JSON_TYPE if json_type is None else json_type
        load_path = self.cfg.JSON_PATH if load_path is None else load_path

        def __json_path_getter(load_path: str) -> list:
            """
            Summary
            -------
                get the json paths from txt or folder

            Parameters
            ----------
                json_type: str
                    choose from ["txt", "folder"]
                load_path: str
                    path of the jsons

            Returns
            -------
                list: list of json pahts
                
            """

            if json_type.lower() == "txt":
                json_paths = list(open(load_path, "r"))

            elif json_type.lower() == "folder":
                json_paths = [load_path + "/" + _ for _ in os.listdir(load_path)]

            else:
                self.logger.critical("Make sure the json_type is chosen from ['txt', 'folder'], you could suggest more json_type to be added")               
            
            return json_paths


        def __json_reader(json_paths: list) -> None:
            """
            Summary
            -------
                read each json object, pass the object to the json extractor
                
            Parameters
            ----------
                json_paths: list
                    list of json paths
                    
            """

            combined_lst = []
            def worker(_):
                json_path = _.strip()
                with open(json_path) as json_obj:
                    try:
                        json_info = json.load(json_obj)
                        json_lst = self.json_extractor(json_info, json_path)
                        combined_lst.extend(json_lst)
                    except json.JSONDecodeError:
                        self.logger.error("JSONDECODEERROR: %s" % json_path)
                    
            with ThreadPool(processes = 40) as pool:
                list(tqdm(pool.imap(worker, json_paths), total=len(json_paths), desc='DataFrame Loading'))
                pool.terminate()
                    
            self.df = pd.DataFrame(combined_lst)
            self.add_columns([self.df.index], ["emb_id"])
            try:
                self.df = self.df.set_index("index_list")
            except KeyError:
                self.logger.error("Perhaps the DataFrame Contains No Data")
                sys.exit(-1)
            
        self.logger.critical("DataFrame Loading Started: %s" % load_path)
        json_paths = __json_path_getter(load_path)
        
        MAX_DF_LEN = 400000
        DF_NUMBER = int(len(json_paths) / MAX_DF_LEN) + 1
        self.logger.debug("Split to %d DataFrame to Loading" % DF_NUMBER)
        temp_dir = "/root/data_hospital_data/temp_dataframes"
        os.makedirs(temp_dir, exist_ok=True)
        def group_elements(n, iterable, padvalue=None):
            return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)
        
        group_items = group_elements(MAX_DF_LEN, json_paths)
        self.total_df_number = 0
        for ind, lst in tqdm(enumerate(group_items), total=DF_NUMBER, desc="DataFrames Loading"):
            clean_lst = [_ for _ in lst if _ is not None]
            __json_reader(clean_lst)
            self.save_to_pickle("%s/%d.pkl" % (temp_dir, ind))
            self.total_df_number = ind + 1
            
        if self.total_df_number == 0:
            self.logger.error("Check Path: %s" % load_path)
            sys.exit(-1)
        
        total_df_lst = []
        for df_ind in tqdm(range(self.total_df_number), desc="Concatenating DataFrames"):
            total_df_lst.append(load_from_pickle("%s/%d.pkl" % (temp_dir, df_ind)))
        self.df = pd.concat(total_df_lst)
        print(self.df)
            
        self.logger.info("DataFrame Loading Finished")
        self.logger.info("DataFrame shape is: %s" % str(self.df.shape))
    

    def save_to_pickle(self, save_path: str, info_dict: dict = None) -> None:
        """
        Summary
        -------
            Save the dataframe and the info of clustering into pickle
            
        Parameters
        ----------
            save_path: str
                the path to save the pickle
            info_dict (optional): dict, default = "None" 
                the info of the clustring
                
        """
        
        if save_path:
            os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)
        try:
            try:
                info_dict = load_from_pickle(save_path, "info") if info_dict is None else info_dict
            except:
                info_dict = None

            pickle_obj = {"info": info_dict, "df": self.df}
            with open(save_path, "wb") as pickle_file: 
                pickle.dump(pickle_obj, pickle_file)
            self.logger.debug("DataFrame Saved: %s" % save_path)    
        except:
            self.logger.warning("DataFrame Saving Failed")
        

    def reset_emb_id(self, ) -> None:
        """
        Summary
        -------
            Reset the emb id, always used after merge dataframe
            
        """
        
        self.add_columns([[_ for _ in range(len(self.df))]], ["emb_id"])


    def merge_dataframe_cols(self, new_columns: pd.DataFrame, merge_style: str) -> None:
        """
        Summary
        -------
            merge new columns to the original self DataFrame
            
        Parameters
        ----------
            new_columns: pd.DataFrame
                the new columns to be added in the format of pd.DataFrame based on different keys.
            merge_style: str
                Merge style include {'left', 'right', 'outer', 'inner', 'cross'}
                left: use only keys from left frame, similar to a SQL left outer join; preserve key order.
                right: use only keys from right frame, similar to a SQL right outer join; preserve key order.
                outer: use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically.
                inner: use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys.
                cross: creates the cartesian product from both frames, preserves the order of the left keys.
                
        """
        self.df = self.df.merge(new_columns, how=merge_style, left_index=True, right_index=True)
    

    def add_columns(self, cols: list, col_labels: list) -> None:
        for i, each in enumerate(cols):
            self.df[col_labels[i]] = each          
            

    def delete_columns(self, delete_key_lst: list) -> None:
        self.df = self.df.drop(delete_key_lst, axis=1)


    def add_rows(self, new_rows: pd.DataFrame) -> None:
        self.df = self.df.append(new_rows)


    def delete_rows(self, delete_index_lst: list) -> None:
        self.df = self.df.drop(delete_index_lst, axis=0)


    def data_updater(self, index, column, new_value) -> None:
        # update certain column
        self.df.at[index, column] = new_value


    def convert_to_json(self, ) -> json:
        return self.df.to_json(orient="index")
        

def merge_dataframe_rows(one_df: pd.DataFrame, another_df: pd.DataFrame) -> DataManager:
    """
    Summary
    -------
        Merge two dataframes by rows, add None to diff columns
        
    Parameters
    ----------
        one_df: pd.DataFrame
        another_df: pd.DataFrame
    
    Returns
    -------
        DataManager: the DataManager instance of combined dataframe
    
    """
    
    temp_ade = DataManager(pd.concat([one_df, another_df], axis=0))
    temp_ade.reset_emb_id()
    print(temp_ade)
    return temp_ade


def load_from_pickle(load_path: str, dforinfo: str = "df") -> pd.DataFrame or dict:
    """
    Summary
    -------
        Load info dict of DataFrame instance from pickle
        
    Parameters
    ----------
        load_path: str
            the path to load pickle, either DataFrame of dict
        dforinfo (optional): str, default = "df"
            either "df" or "info" for now
        
    Returns
    -------
        pd.DataFrame of dict: depends on dforinfo
        
    """
    
    with open(load_path, "rb") as pickle_file:
        return pickle.load(pickle_file)[dforinfo]
    
