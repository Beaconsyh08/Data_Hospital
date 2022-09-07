
# train_path = "/data_path/v40_train.txt"
import json
import os
import pickle
from collections import Counter, defaultdict
from multiprocessing.pool import ThreadPool

import pandas as pd
from src.utils.logger import get_logger
from configs.config import LogicalCheckerConfig
from src.data_manager.data_manager_creator import data_manager_creator
from tqdm import tqdm


class DuplicatedDoctor():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.ROOT_PATH = "%s/duplicated_doctor" % cfg.ROOT
        os.makedirs(self.ROOT_PATH, exist_ok=True)
        self.logger = get_logger()

        self.NEW_DATA_PATH = cfg.JSON_PATH
        self.ORI_DATA_PATH = cfg.ORI_JSON_PATH
        self.PKL_READY = cfg.PKL_READY
        self.ORI_PKL_PATH = cfg.ORI_PKL_PATH
        self.SAVE_PKL_PATH = cfg.SAVE_PKL_PATH
        self.stats_dict = dict()
        self.to_del_paths = []
        self.method = cfg.METHOD.lower()
        
        self.json_paths_ori = [_.strip() for _ in list(open(self.ORI_DATA_PATH, "r"))]
        if self.method == "incremental":
            self.json_paths_new = [_.strip() for _ in list(open(self.NEW_DATA_PATH, "r"))]
        else:
            self.json_paths_new = []
            
        self.json_paths = self.json_paths_ori + self.json_paths_new
        
        self.nodup_jsons = self.json_duplicated()
        self.nodup_jsons_new = self.nodup_jsons - set(self.json_paths_ori) - set(self.dup_jsons)
        
        if self.PKL_READY:
            self.ORI_DATA = pd.read_pickle(self.ORI_PKL_PATH)
        else:
            res_dict = defaultdict(list)            
            self.ORI_DATA = self.map_loader(res_dict, self.nodup_jsons)
            self.save_to_pickle(self.ORI_DATA, self.SAVE_PKL_PATH)
        
        if self.method == "total" or len(self.nodup_jsons_new) == 0:
            self.TOTAL_DATA = self.ORI_DATA
        else:            
            self.TOTAL_DATA = self.map_loader(self.ORI_DATA, self.nodup_jsons_new)    
            self.save_to_pickle(self.ORI_DATA, self.SAVE_PKL_PATH)
                
    
    def json_duplicated(self,) -> set:
        nodup_paths = set(self.json_paths)
        self.dup_jsons = [_ for _, count in Counter(self.json_paths).items() if count > 1]
        self.to_del_paths += self.dup_jsons
        self.logger.debug("Duplicated Json Number: %d" % len(self.dup_jsons))
        self.stats_dict["dup_json"] = len(self.dup_jsons)
        
        return nodup_paths
    
                    
    def map_loader(self, res_dict: defaultdict, json_paths: list) -> dict:
        def worker(_):
            json_path = _.strip()
            
            with open(json_path) as json_obj:
                json_info = json.load(json_obj)
                try:
                    img_url = json_info["imgUrl"]
                except:
                    print(json_path)
                res_dict[img_url].append(json_path)

        with ThreadPool(processes = 40) as pool:
            list(tqdm(pool.imap(worker, json_paths), total=len(json_paths), desc='Map Loading'))
            pool.terminate()
            
        return res_dict
    

    def save_to_pickle(self, pickle_obj: dict, save_path: str) -> None:
        with open(save_path, "wb") as pickle_file: 
            pickle.dump(pickle_obj, pickle_file)
            
    
    def self_duplicated_finder(self, img_json_dict: dict, res_stats: dict) -> dict:
        dup_dict = defaultdict(list)
        dup_img_no = 0        
        for key, value in tqdm(img_json_dict.items(), desc="Duplicated Checking"):
            if len(value) > 1:
                dup_dict[key] = value
                dup_img_no += len(value) - 1
                
        self.logger.debug("Duplicated Img Number: %d" % dup_img_no)
        res_stats["dup_img"] = dup_img_no
        return dup_dict     
        
    
    def time_checker(self, dup_dict: dict) -> dict:
        to_del_paths = []
        def worker(_):
            mod_dict = dict()
            for v in _:
                mod_time = os.path.getmtime(v)
                mod_dict[v] = mod_time
                
            if self.method == "total":
                sorted_mod_dict = dict(sorted(mod_dict.items(), key=lambda item: item[1]))
                for _ in list(sorted_mod_dict.keys())[:-1]:
                    to_del_paths.append(_)
            elif self.method == "inference":
                for _ in list(mod_dict.keys()):
                    to_del_paths.append(_)

        with ThreadPool(processes = 40) as pool:
            list(tqdm(pool.imap(worker, list(dup_dict.values())), total=len(list(dup_dict.values())), desc='ImgUrl Searching'))
            pool.terminate()
        
        self.to_del_paths += to_del_paths
    
    
    def save_results(self, ) -> None:
        with open ("%s/duplicated.txt" % self.ROOT_PATH, "w") as to_del_file:
            for json_path in self.to_del_paths:
                to_del_file.writelines(json_path + "\n")
                
        with open("%s/clean.txt" % self.ROOT_PATH, "w") as clean_file:
            with open (self.ORI_DATA_PATH) as ori_file:
                ori_jsons = [_.strip() for _ in ori_file]
                if self.method == "total":
                    clean_jsons = (set(ori_jsons) - set(self.to_del_paths)) | set(self.dup_jsons)
                elif self.method == "inference":
                    clean_jsons = (set(ori_jsons) - set(self.to_del_paths))
                for json_path in clean_jsons:
                    clean_file.writelines(json_path + "\n")
                    
        print("Duplicted Stats", self.stats_dict)
                
                
    def build_logical_df(self, ):
        dm = data_manager_creator(LogicalCheckerConfig)
        dm.load_from_json()
        dm.save_to_pickle(self.cfg.LOGICAL_DATAFRAME_PATH)
        
        
    def self_diagnose(self,) -> None:
        dup_dict = self.self_duplicated_finder(self.ORI_DATA, self.stats_dict)
        self.time_checker(dup_dict)
        self.save_results()
        self.build_logical_df()


if __name__ == '__main__':
    NAME = "add_matching_2"
    class Config:
        # ROOT = '/share/analysis/result/data_hospital/0628/%s' % NAME
        ROOT = '/root/data_hospital/0728v60/%s' % NAME
        LOGICAL_DATAFRAME_PATH = '%s/dataframes/logical_dataframe.pkl' % ROOT
        REPROJECT_DATAFRAME_PATH = '%s/dataframes/reproject_dataframe.pkl' % ROOT
        FINAL_DATAFRAME_PATH = '%s/dataframes/final_dataframe.pkl' % ROOT
    class DuplicatedDoctorConfig(Config):
        JSON_PATH = "/data_path/%s.txt" % NAME
        ORI_JSON_PATH = "/data_path/%s.txt" % NAME
        ORI_PKL_PATH = "/root/data_hospital/dataframes/sidecam_ori.pkl"
        SAVE_PKL_PATH = "/root/data_hospital/dataframes/%s.pkl" % NAME
        PKL_READY = False
        METHOD = "inference"
        
    dd = DuplicatedDoctor(DuplicatedDoctorConfig)
    dd.self_diagnose()
