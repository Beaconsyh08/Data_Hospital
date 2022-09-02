import pandas as pd    
import threading
import os 
from src.utils.struct import Obstacle, parse_obs
from pathlib import Path
from src.visualization.visualization import Visualizer
from configs.config import VisualizationConfig
from tqdm import tqdm
import json
import pickle
from src.utils.logger import get_logger


class MatchingDoctor():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.logger = get_logger()
        self.df = pd.read_pickle(self.cfg.BADCASE_DATAFRAME_PATH_DMISS)        
        self.logger.info("Bad Case DataFrame Loaded")
        
        self.save_dir = cfg.SAVE_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.vis = cfg.VIS
        self.new_df_path = cfg.BADCASE_DATAFRAME_PATH_DMATCH
            
            
    def save_to_pickle(self, pickle_obj: dict, save_path: str) -> None:
        with open(save_path, "wb") as pickle_file: 
            pickle.dump(pickle_obj, pickle_file)
            
            
    def result_output(self, jsons: set, save_name: str):
        with open("%s/%s.txt" % (self.save_dir, save_name), "w") as to_del_file:
            for json_path in tqdm(jsons):
                with open(json_path) as f:
                    json_obj = json.load(f)
                    to_del_file.writelines(json_obj["ori_path"] + "\n")
                    
    
    def rules_decider(self,) -> set:
        df_miss = self.df[((self.df['flag'] == 'miss') & (self.df['case_flag']=='0')) & (self.df['priority'] == 'P0')]
        self.df_selected = df_miss[(df_miss.dis_ratio > 0.5) & (df_miss.euclidean_dis_diff > 5)]
        
        matching_jsons = set(self.df_selected.json_path)
        self.logger.debug("2D3D Matching Frame Number: %d" % len(matching_jsons))
        
        new_df = self.df[~self.df.json_path.isin(matching_jsons)]
        self.save_to_pickle(new_df, self.new_df_path)
        
        return matching_jsons
    
    
    def diagnose(self,):
        matching_jsons = self.rules_decider()
        self.result_output(matching_jsons, "to_del")    
        total_jsons = set(self.df.json_path)
        clean_jsons = total_jsons - matching_jsons
        self.result_output(clean_jsons, "clean")
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
    
