from pathlib import Path
from configs.config import VisualizationConfig
from src.utils.struct import Obstacle, parse_obs
from src.visualization.visualization import Visualizer
import pandas as pd
import threading
import os
import pickle
from tqdm import tqdm
import json
from src.utils.logger import get_logger


class MissAnnoChecker():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.logger = get_logger()
        self.df = pd.read_pickle(self.cfg.BADCASE_DATAFRAME_PATH)["df"]        
        self.logger.info("Bad Case DataFrame Loaded")
        
        self.save_dir = cfg.SAVE_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        self.vis = cfg.VIS
        self.new_df_path = cfg.BADCASE_DATAFRAME_PATH_DMISS
        
        
    def rules_decider(self,) -> set:
        self.df_false_p0 = self.df[(self.df.priority == "P0") & (self.df.case_flag == "2") & (self.df.iou == -1)]
        false_p0_jsons = set(self.df_false_p0.json_path)
        self.logger.debug("Miss Label P0 Frame Number: %d" % len(false_p0_jsons))
        new_df = self.df[~self.df.json_path.isin(false_p0_jsons)]
        self.save_to_pickle(new_df, self.new_df_path)
        return false_p0_jsons
    
    
    def save_to_pickle(self, pickle_obj: dict, save_path: str) -> None:
        with open(save_path, "wb") as pickle_file: 
            pickle.dump(pickle_obj, pickle_file)
            
            
    def result_output(self, jsons: set, save_name: str):
        with open("%s/%s.txt" % (self.save_dir, save_name), "w") as to_del_file:
            
            for json_path in tqdm(jsons):
                with open(json_path) as f:
                    json_obj = json.load(f)
                    to_del_file.writelines(json_obj["ori_path"] + "\n")
                    
    
    def diagnose(self,):
        false_p0_jsons = self.rules_decider()
        self.result_output(false_p0_jsons, "miss_anno_error")    
        total_jsons = set(self.df.json_path)
        clean_jsons = total_jsons - false_p0_jsons
        self.result_output(clean_jsons, "clean")
        if self.vis:
            self.visualization(self.df_false_p0.sample(500))
    
    
    def visualization(self, df_vis: pd.DataFrame):
        visualizer = Visualizer(VisualizationConfig)
        for idx, row in tqdm(enumerate(df_vis.itertuples()), total=len(df_vis)):
            obs_list = []
            class_name, id = row.class_name, row.id
            obs = parse_obs(row)
            obs_list.append(obs)
            
            save_dir = "%s/%s/%s_%d" % (self.save_dir, class_name, str(Path(row.json_path).stem), id)
            save_path = '%s.jpg' % save_dir
            t0 = threading.Thread(target=visualizer.draw_bbox_0, args=(obs_list, save_path))
            t0.start()

        
if __name__ == '__main__':
    pass
    
