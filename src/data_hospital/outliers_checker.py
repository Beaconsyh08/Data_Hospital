from src.data_manager.data_manager import load_from_pickle
from src.data_manager.data_manager_creator import data_manager_creator
from src.data_manager.train_cam3d_manager import TrainCam3dManager
from src.utils.logger import get_logger
import numpy as np
import pandas as pd 

class OutlierChecker():
    def __init__(self, cfg: dict) -> None:
        self.logger = get_logger()
        self.cfg = cfg       
        self.dm = self.build_df()
        self.df = self.dm.df
        print(self.df)
    
    
    def build_df(self, ) -> TrainCam3dManager:
        dm = data_manager_creator(self.cfg)
        dm.load_from_json()
        return dm
    
    
    def iqr(self,):
        selected_attrbute = ["yaw", "x", "y"]
        print(self.df)
        selected_ori = self.df[self.df.camera_orientation == "front_left_camera"]
        print(selected_ori)
        selected_df = selected_ori[selected_attrbute].dropna()
        print(selected_df)
        result = selected_df.apply(self.find_iqr)
        print(result)
        
        for each in selected_attrbute:
            upper = result.at[0, each]
            lower = result.at[1, each]
            
            outlier_df = selected_df[(selected_df[each] > upper) | (selected_df[each] < lower)]
            print("--------------------------------------------------------")
            print(each)
            print(outlier_df)
            
            if each == "x":
                print("xxx")
                print(selected_df[(selected_df[each] < 0)])
                
            if each == "y":
                print("yyy")
                print(selected_df[(selected_df[each] > 0)])

            print("--------------------------------------------------------")
            
        
    def find_iqr(self, x):
        q3, q1 = np.percentile(x, [75, 25])
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 *iqr
        return upper, lower, max(x), min(x)
        
    
    def sd(self,):
        pass
    
    
    def diagnose(self) -> None:
        self.iqr()
        