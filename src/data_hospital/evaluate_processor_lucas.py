import sys
from time import sleep
from datetime import datetime

from configs.config_data_hospital import DataHospitalConfig, EvaluateProcessorConfig
from src.data_manager.data_manager_creator import data_manager_creator

import json
import os

import requests
from src.utils.logger import get_logger
from tqdm import tqdm


class EvaluateProcessorLucas():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.logger = get_logger()
        self.dt_dir = "%s/dt/" % cfg.INPUT_DIR
        self.logger.debug("DT_PATH: %s" % self.dt_dir)
        self.gt_dir = "%s/gt/" % cfg.INPUT_DIR
        self.logger.debug("GT_PATH: %s" % self.gt_dir) 
        self.output_dir = cfg.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.dt_path = "%s/dt.txt" % self.output_dir
        self.gt_path = "%s/gt.txt" % self.output_dir
        self.result_path = "%s/result.json" % self.output_dir
        self.badcases_dir = "%s/cases" % self.output_dir
        
    
    def path_walker(self, parent_dir: str) -> list:
        file_lst = []
        for root, dirs, files in os.walk(parent_dir):
            for file in files:
                file_name = os.path.join(root, file)
                file_lst.append(file_name)
                    
        return file_lst
    
    
    def save_results(self, save_path:str, save_dir:list) -> None:
        with open (save_path, "w") as output_file:
            for path in save_dir:
                output_file.writelines(path + "\n")
            
    
    def pre_process(self, ):
        dt_lst = self.path_walker(self.dt_dir)
        gt_lst = self.path_walker(self.gt_dir)
        self.save_results(self.dt_path, dt_lst)
        self.save_results(self.gt_path, gt_lst)
    
    
    def evaluate(self,):
        # execute("a", "b", "c")
        os.system("cd ../Lucas_Evaluator && python exec_eva/exec_obstacle_2d_eva.py --gt_dir %s --dt_dir %s --case_dir %s --result_dir %s" % (self.gt_path, self.dt_path, self.badcases_dir, self.result_path))
        return 
        
    
    def build_badcase_df(self, ) -> None:
        dm = data_manager_creator(self.cfg)
        dm.load_from_json()
        dm.save_to_pickle(self.cfg.BADCASE_DATAFRAME_PATH)
    
    
    def diagnose(self, ):
        self.pre_process()
        self.evaluate()
        badcases_lst = self.path_walker(self.badcases_dir)
        self.save_results(self.cfg.JSON_PATH, badcases_lst)
        self.build_badcase_df()
        
        
if __name__ == '__main__':
    e = EvaluateProcessorLucas(EvaluateProcessorConfig)
    e.diagnose()
    