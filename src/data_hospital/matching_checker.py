from re import match
import pandas as pd    
import threading
import os
from src.data_manager.data_manager import load_from_pickle 
from src.utils.struct import Obstacle, parse_obs
from pathlib import Path
from src.visualization.visualization import Visualizer
from configs.config import VisualizationConfig, DataHospitalConfig
from tqdm import tqdm
import json
import pickle
from src.utils.logger import get_logger


class MatchingChecker():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.logger = get_logger()
        self.df_path = self.cfg.BADCASE_DATAFRAME_PATH
        self.df = load_from_pickle(self.df_path)
        self.logger.info("Bad Case DataFrame Loaded")
        self.save_dir = cfg.SAVE_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        self.vis = cfg.VIS
            
            
    def save_to_pickle(self, pickle_obj: dict, save_path: str) -> None:
        with open(save_path, "wb") as pickle_file: 
            pickle.dump(pickle_obj, pickle_file)
            
            
    def result_output(self, jsons: set, save_name: str) -> None:
        with open("%s/%s.txt" % (self.save_dir, save_name), "w") as to_del_file:
            if jsons != {None}:
                for json_path in tqdm(jsons):
                    to_del_file.writelines(str(json_path) + "\n")
                    
    
    def rules_decider(self,) -> set:
        df_miss = self.df[((self.df['flag'] == 'miss') & (self.df['case_flag']=='0')) & (self.df['priority'] == 'P0')]
        self.df_selected = df_miss[(df_miss.dis_ratio > 0.5) & (df_miss.euclidean_dis_diff > 5)]
        
        self.df_selected = self.df_selected.assign(matching_error=1)
        self.matching_dict = self.df_selected.matching_error.to_dict()
        self.df['matching_error'] = [self.matching_dict.get(_, 0) for _ in self.df.index]
        
        matching_jsons = set(self.df_selected.ori_path) if DataHospitalConfig.COOR == "Lidar" else set(self.df_selected.json_path)
        self.logger.debug("2D3D Matching Frame Number: %d" % len(matching_jsons))
        
        self.save_to_pickle(self.df, self.df_path)
        
        return matching_jsons
    
    
    def save_result_to_ori_df(self,):
        self.df_selected.index = [_[:-3] for _ in list(self.df_selected.index)]   
        matching_dict = self.df_selected.matching_error.to_dict()
        ori_df = load_from_pickle(self.cfg.DATAFRAME_PATH)
        ori_df["matching_error"] = [matching_dict.get(_, 0) for _ in ori_df.index]
        self.save_to_pickle(ori_df, self.cfg.DATAFRAME_PATH)
        self.logger.debug("Update Result to Ori DataFrame: %s" % self.cfg.DATAFRAME_PATH)
    
    
    def diagnose(self,):
        matching_jsons = self.rules_decider()
        self.result_output(matching_jsons, "matching_error")    
        total_jsons = set(self.df.ori_path) if DataHospitalConfig.COOR == "Lidar" else set(self.df.json_path)
        clean_jsons = total_jsons - matching_jsons
        self.result_output(clean_jsons, "clean")
        self.save_result_to_ori_df()
        if self.vis:
            self.visualization(self.df_selected.sample(500))
        
        
    def visualization(self, df_vis: pd.DataFrame):
        visualizer = Visualizer(VisualizationConfig)
        for idx, row in tqdm(enumerate(df_vis.itertuples()), total=len(df_vis)):
            obs_list = [parse_obs(row)]
            dis_ratio, euclidean_dis_diff = round(row.dis_ratio, 2), round(row.euclidean_dis_diff, 2)
            
            save_dir = "%s/images/%d_%s_%s_%s" % (self.save_dir, i, str(dis_ratio), str(euclidean_dis_diff), row.camera_orientation)
            save_path = '%s.jpg' % save_dir
            bev_save_path = '%s_bev.jpg' % save_dir
            
            if not pd.isna(row.peer_id) and row.peer_id in self.df.index:
                _ = self.df.loc[row.peer_id]
            peer_row = _ if type(_) == pd.Series else _.iloc[0]
            obs_list.append(parse_obs(peer_row))
                
            t0 = threading.Thread(target=visualizer.draw_bbox_0, args=(obs_list, save_path))
            t0.start()
            t1 = threading.Thread(target=visualizer.plot_bird_view, args=(obs_list, bev_save_path))
            t1.start()
        
                    
if __name__ == '__main__':
    pass
    
