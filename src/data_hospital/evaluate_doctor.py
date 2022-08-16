import sys
from time import sleep
from datetime import date
sys.path.append("../haomo_ai_framework")
import json
import os
from itertools import zip_longest

import requests
from haomoai.cards import CardOperation
from src.utils.logger import get_logger
from tqdm import tqdm


class EvaluateDoctor():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.logger = get_logger()
        self.dt_path = "%s/dt/" % cfg.INPUT_DIR
        self.logger.debug("DT_PATH: %s" % self.dt_path)
        self.gt_path = "%s/gt/" % cfg.INPUT_DIR
        self.logger.debug("GT_PATH: %s" % self.gt_path) 
        self.result_flag = False
        self.request_id = "%s_%s" % (cfg.NAME, str(date.today()))
        self.output_dir = cfg.OUTPUT_DIR
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
                "sensor": "front_left_camera",
                "tasks": self.eval_data
                }

        r = requests.post(url="http://10.100.4.203:5000/evaluate_lucas", json=data)
        
        self.logger.info("Evaluating")
        return r.json()


    def get_result(self,):
        data = {"request_id": self.request_id, "whitelist": "", "sceneFile": ""}

        r = requests.post(url="http://10.100.4.203:5000/get_lucas_evaluate_result", json=data)
        
        self.logger.info("Getting Result")
        return r.json()
        
    
    def diagnose(self, ):
        dt_card = self.card_generator_json("icu30", "dt", self.dt_path)
        gt_card = self.card_generator_json("icu30", "gt", self.gt_path)
        self.eval_data = [{"gt_card": gt_card, "dt_card":dt_card}]
        
        self.evaluate()
        while not self.result_flag:
            result = self.get_result()
            self.logger.info(result["desc"])
            if result["desc"].endswith("2"):
                self.result_flag = True
                self.logger.critical("Evaluate Completed")
                break
            sleep(600)
        
        save_path = "%s/result.json" % self.output_dir
        with open(save_path, 'w') as output_file:
            json.dump(result, output_file)
            
        self.logger.debug("Result Has Been Saved in: %s" % save_path)