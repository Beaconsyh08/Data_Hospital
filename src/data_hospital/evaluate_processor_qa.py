import sys
from time import sleep
from datetime import datetime
import yaml

from configs.config_data_hospital import DataHospitalConfig
from src.data_manager.data_manager_creator import data_manager_creator
sys.path.append("../haomo_ai_framework")
import json
import os

import requests
from haomoai.cards import CardOperation
from src.utils.logger import get_logger
from tqdm import tqdm


class EvaluateProcessorQA():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.logger = get_logger()
        self.dt_dir = "%s/dt/" % cfg.INPUT_DIR
        self.logger.debug("dt_dir: %s" % self.dt_dir)
        self.gt_dir = "%s/gt/" % cfg.INPUT_DIR
        self.logger.debug("gt_dir: %s" % self.gt_dir) 
        self.result_flag = False
        self.request_id = "%s_%s_%s" % (cfg.MODEL_NAME, cfg.NAME, str(datetime.now()))
        self.output_dir = cfg.OUTPUT_DIR
        if DataHospitalConfig.ORIENTATION == "SIDE":
            self.sensor = "front_left_camera"
        elif DataHospitalConfig.ORIENTATION == "FRONT":
            self.sensor = "front_middle_camera"
        os.makedirs(self.output_dir, exist_ok=True)
        
        
    def card_generator_json(self, project: str, media_name: str, dir: list):
        oss_urls = []
        card_inst = CardOperation()

        for _, _, files in os.walk(dir):
            for name in tqdm(files):
                js_name = dir + '/' + name
                with open(js_name, 'r', encoding='utf-8') as f:
                    js_str = str(json.load(f))
                    oss_urls.append(js_str)

        card_id = card_inst.create_card_w_append(project=project, media_name=media_name, target_dir=dir) # 设置生成卡片的 project 和 media_name
        
        res = {"project": project, "id": card_id, "name": media_name}
        self.logger.info(res)
        return res
    
    
    def evaluate(self,):
        data = {"request_id": self.request_id,
                "project_name": "icu30",
                "algorithm_type": "perception",
                "model_type": "model",
                "model_name": "2d_obstacle",
                "sensor": self.sensor,
                "tasks": self.eval_data
                }

        r = requests.post(url="http://10.100.4.203:5000/evaluate_lucas", json=data)
        
        self.logger.info("Evaluating")
        self.logger.info(data)
        return r.json()


    def get_result(self,):
        data = {"request_id": self.request_id, "whitelist": "", "sceneFile": ""}

        r = requests.post(url="http://10.100.4.203:5000/get_lucas_evaluate_result", json=data)
        
        self.logger.info("Getting Result")
        return r.json()
        
    
    def append_2_yaml(self, badcase_card: dict) -> None:
        new_yaml = {'file_name': self.cfg.JSON_PATH, 
                    'cards': [{'card_id': badcase_card["id"], 'media_name': "cases", 'project': badcase_card["project"]}]}
        yaml_path = "dataset/dataset.yaml"
        new_yaml_path = "dataset/new_dataset.yaml"
            
        with open(yaml_path) as input_file:
            data = yaml.safe_load(input_file)
            data['cards_conf'].append(new_yaml)
            
        with open(new_yaml_path, 'w') as output_file:
            yaml.dump(data, output_file)
        
        os.system("./run.sh -d %s" % new_yaml_path) 
        
    
    def build_badcase_df(self, ) -> None:
        dm = data_manager_creator(self.cfg)
        dm.load_from_json()
        dm.save_to_pickle(self.cfg.BADCASE_DATAFRAME_PATH)
    
    
    def diagnose(self, ):
        dt_card = self.card_generator_json("icu30", "dt", self.dt_dir)
        gt_card = self.card_generator_json("icu30", "gt", self.gt_dir)
        self.eval_data = [{"gt_card": gt_card, "dt_card": dt_card}]
        
        self.evaluate()
        while not self.result_flag:
            result = self.get_result()
            self.logger.info(result["desc"])
            if result["desc"].endswith("2"):
                self.result_flag = True
                self.logger.critical("Evaluate Completed")
                break
            elif result["desc"].endswith("x"):
                self.logger.error("QA Evaluator Not Available, Contact QA")
                sys.exit(0)
            sleep(600)
        
        save_path = "%s/result.json" % self.output_dir
        with open(save_path, 'w') as output_file:
            json.dump(result, output_file)
            
        self.logger.debug("Result Has Been Saved in: %s" % save_path)
        
        badcase_card = result["data"]["badcase_card"]
        self.append_2_yaml(badcase_card)
        self.build_badcase_df()
        
        
if __name__ == '__main__':
    class Config1:
        NAME = "train31_total"
        INPUT_DIR = '%s/data_inferencer/'% '/root/data_hospital_data/0728v60/%s'
        OUTPUT_DIR = '%s/evaluate_processor' % '/root/data_hospital_data/0728v60/%s'
        MODEL_NAME = "HAHA"
        
        JSON_PATH = '/data_path/data_hospital_badcase.txt'
        JSON_TYPE = "txt"
        DATA_TYPE = "qa_cam3d_temp"
        ROOT = '/root/data_hospital_data/0728v60/%s' % NAME
        BADCASE_DATAFRAME_PATH = '%s/dataframes/badcase_dataframe.pkl' % ROOT
    
    evaluator = EvaluateProcessorQA(Config1)
    # evaluator.eval_data = [{"gt_card": {'project': 'icu30', 'id': '62d173f328f4f2a8671b2874', 'name': 'trans_zhouping_side_gt'}, 
    #                         "dt_card":{'project': 'qa', 'id': '62e90c382f748ee9edbfb0f6', 'name': '0804_SIDECAM'}}]
    # evaluator.evaluate()
    evaluator.request_id = '/share/analysis/syh/models/2.2.4.0-0714-M-PT230-288-U_yayun_night_2022-09-02 20:53:48.959431'
    result = evaluator.get_result()
    print(result)
    
    # save_path = "%s/result.json" % Config1.OUTPUT_DIR
    # with open(save_path, 'w') as output_file:
    #     json.dump(result, output_file)
        
    # print("Result Has Been Saved in: %s" % save_path)
    
    # badcase_card = result["data"]["badcase_card"]
    # evaluator.append_2_yaml(badcase_card)
    # evaluator.build_badcase_df()
        
    