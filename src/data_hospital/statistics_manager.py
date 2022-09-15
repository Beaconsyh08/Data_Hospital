import os

import folium
import matplotlib.pyplot as plt
from folium.plugins import HeatMap, MarkerCluster
from configs.config_data_hospital import DataHospitalConfig, DuplicatedCheckerConfig, LogicalCheckerConfig, Config, CalibrationCheckerConfig
from src.data_manager.data_manager import load_from_pickle
from src.data_manager.data_manager_creator import data_manager_creator
from src.utils.logger import get_logger
import sys
sys.path.append("../haomo_ai_framework")
from haomoai.cards import CardOperation
from tqdm import tqdm

plt.rc('font', size=16)
plt.rc('axes', titlesize=24) 
plt.rc('axes', labelsize=20)
from datetime import time
import pandas as pd


class StatisticsManager():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.df = load_from_pickle(self.cfg.DATAFRAME_PATH)
        self.total_frames_number = len(set(list(self.df.json_path)))
        self.df['class_name_map'] = self.df['class_name'].map(Config.TYPE_MAP)
        self.save_dir = self.cfg.SAVE_DIR
        self.logger = get_logger()
        os.makedirs(self.save_dir, exist_ok=True)
        self.checking_stats()


    def addlabels(self, x, y):
        for i in range(len(x)):
            plt.text(i, y[i], y[i], ha = 'center')
        
    
    def set_loader(self, load_path: str):
        return set([_.strip() for _ in open(load_path)])
        
    
    def checking_stats(self,):
        dup_jsons = self.set_loader("%s/duplicated_jsons.txt" % DuplicatedCheckerConfig.SAVE_DIR)
        dup_imgs = self.set_loader("%s/duplicated_imgs.txt" % DuplicatedCheckerConfig.SAVE_DIR)
        ori_empty = self.set_loader("%s/empty.txt" % LogicalCheckerConfig.SAVE_DIR)
        checked_jsons = set(list(self.df.json_path))
        real_empty_frame = ori_empty - checked_jsons
        
        problem_frame, problem_bbox = set(), set()
        total_frame = checked_jsons | dup_jsons | dup_imgs | ori_empty
        total_frame_amount = len(total_frame) + len(dup_jsons)
        problem_frame = dup_jsons | dup_imgs | real_empty_frame
        print(total_frame_amount, len(dup_jsons), len(dup_imgs), len(ori_empty), len(checked_jsons))
        
        res_pd = pd.DataFrame()
        res_pd.set_axis(DataHospitalConfig.PROBLEMATIC_LIST + DataHospitalConfig.ERROR_LIST, inplace=True)
        
        res_pd.at["dup_json", "Frame_Amount"], res_pd.at["dup_json", "Frame_Ratio"] = len(dup_jsons), len(dup_jsons) / total_frame_amount
        res_pd.at["dup_img", "Frame_Amount"], res_pd.at["dup_img", "Frame_Ratio"] = len(dup_imgs), len(dup_imgs) / total_frame_amount
        res_pd.at["empty", "Frame_Amount"], res_pd.at["empty", "Frame_Ratio"] = len(real_empty_frame), len(real_empty_frame) / total_frame_amount
        
        total_instance_amount = len(self.df)
        for error in DataHospitalConfig.ERROR_LIST:
            error_df = self.df[self.df[error] != 0]
            error_instance_amount = len(error_df)
            error_frame = set(list(error_df.json_path))
            problem_frame |= error_frame
            error_frame_amount = len(error_frame)
            res_pd.at[error, "Instance_Amount"], res_pd.at[error, "Instance_Ratio"] = error_instance_amount, error_instance_amount / total_instance_amount
            res_pd.at[error, "Frame_Amount"], res_pd.at[error, "Frame_Ratio"] = error_frame_amount, error_frame_amount / total_frame_amount 
        
        res_pd.at["total_error", "Frame_Amount"], res_pd.at["total_error", "Frame_Ratio"] = len(problem_frame), len(problem_frame) / total_frame_amount
        save_path = "%s/error_stats.xlsx" % self.save_dir
        res_pd.to_excel(save_path, index=True, header=True)
        print(res_pd)
        self.logger.debug("Error Stats Saved in: %s" % save_path)
        
    
    def city_stats(self,):
        frame_df = self.df.groupby(["json_path", "city"]).size().unstack(fill_value=0)
        city_lst = frame_df.columns.to_list()
        city_dict = dict()
        city_ratio_dict = dict()
        
        for city in city_lst:
            city_df = frame_df[frame_df[city] > 0]
            city_dict[city] = len(city_df)
        total_values = sum([val for val in city_dict.values()])
        for key in city_dict:
            city_ratio_dict[key] = city_dict[key]/total_values

        city_dict = dict(sorted(city_dict.items(), key=lambda item: item[1], reverse=True))
        city_ratio_dict = dict(sorted(city_ratio_dict.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame(city_dict, index=[0]).T
        df_ratio = pd.DataFrame(city_ratio_dict, index=[0]).T

        city_keys, city_values = list(city_dict.keys()), list(city_dict.values())
        fig = plt.figure(figsize=(25, 20))
        plt.bar(city_keys, city_values)
        self.addlabels(city_keys, city_values)
        plt.title("Frames By City")
        plt.xlabel("City")
        plt.ylabel("Number of Frames")
        save_path = "%s/city_hist.jpg" % self.save_dir
        plt.savefig(save_path)
        plt.close()
        self.logger.debug("City Hist Saved in %s" % save_path)
        
        save_path = "%s/city_stats_result.xlsx" % self.save_dir

        ### cal ratio
        df_concat = pd.concat([df,df_ratio],axis=1)
        df_concat.set_axis(["Amount", "Ratio"], axis='columns', inplace=True)
        print(df_concat)
        df_concat.to_excel(save_path, index=True, header=True)
        
        self.logger.debug("City Stats Saved in %s" % save_path)
        
        
    def city_heat_distribution(self, ) -> None:
        m = folium.Map([39.904989, 116.405285], tiles='stamentoner', zoom_start=10)
        df = self.df.drop_duplicates(subset="json_path", keep='first', inplace=True)
        df = self.df[self.df["lon"].notna()]
        
        lons = df.lon.to_list()
        lats = df.lat.to_list()
        
        comb = [[lats[i], lons[i], 1] for i in range(len(lats))]
        HeatMap(comb).add_to(m)
        
        save_path = "%s/city_heatmap.html" % self.save_dir
        m.save(save_path)
        self.logger.debug("City Heat Map Saved in: %s" % save_path)
    
    
    def time_stats(self,):
        
        time_dict = dict()
        time_ratio_dict = dict()
        night_df = self.df[(self.df["time"] < time(6, 0, 0)) | (self.df["time"] > time(19, 0, 0))]
        day_df = self.df[(self.df["time"] >= time(6, 0, 0)) & (self.df["time"] <= time(19, 0, 0))]
        night_json = list(set(night_df.json_path.to_list()))
        day_json = list(set(day_df.json_path.to_list()))
        time_dict["night"] = len(night_json)
        time_dict["day"] = len(day_json)
        # self.output_result_txt(night_json, "night")
        
        total_values = sum([val for val in time_dict.values()])
        time_ratio_dict["night"] = time_dict["night"]/total_values
        time_ratio_dict["day"] = time_dict["day"]/total_values
        
        time_dict = dict(sorted(time_dict.items(), key=lambda item: item[1], reverse=True))
        time_ratio_dict = dict(sorted(time_ratio_dict.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame(time_dict, index=[0]).T
        df_ratio = pd.DataFrame(time_ratio_dict, index=[0]).T
        

        time_keys, time_values = list(time_dict.keys()), list(time_dict.values())
        fig = plt.figure(figsize=(25, 20))
        plt.bar(time_keys, time_values)
        self.addlabels(time_keys, time_values)
        plt.title("Frames By Time")
        plt.xlabel("Time")
        plt.ylabel("Number of Frames")
        save_path = "%s/time_hist.jpg" % self.save_dir
        plt.savefig(save_path)
        plt.close()
        self.logger.debug("Time Hist Saved in %s" % save_path)
                
        save_path = "%s/time_stats_result.xlsx" % self.save_dir

        ### cal ratio
        df_concat = pd.concat([df,df_ratio],axis=1)
        df_concat.set_axis(["Amount", "Ratio"], axis='columns', inplace=True)
        print(df_concat)
        df_concat.to_excel(save_path, index=True, header=True)
        self.logger.debug("Time Stats Saved in %s" % save_path)
    
    
    def info_stats(self,):
        for info in ["class_name_map", "camera_orientation"]:
            fig = plt.figure(figsize=(25, 20))
            cat_num = len(self.df[info].unique())

            self.df[info].hist(bins=max(10, cat_num))
            plt.title("Frames By %s" % info.capitalize())
            plt.xlabel(info.capitalize())
            plt.ylabel("Number of Frames")
            if cat_num > 10:
                plt.xticks(rotation = 90)
            
            path = "%s/%s_hist.jpg" % (self.save_dir, info)
            plt.savefig(path)
            plt.close()
            
            self.logger.debug("%s Hist Saved in %s" % (info.capitalize(), path))

            df = pd.concat([self.df[info].value_counts(), self.df[info].value_counts(normalize=True)],axis=1)
            df.set_axis(["Amount", "Ratio"], axis='columns', inplace=True)
            print(df)
            save_path = "%s/%s_stats_result.xlsx" % (self.save_dir, info)
            df.to_excel(save_path, index=True, header=True)
            
            self.logger.debug("%s Stats Saved in %s" % (info.capitalize(), save_path))
            
            
    def scenario_bbox_stats(self,):
        bigcar_df = self.df[(abs(self.df.x) < 8) & (abs(self.df.y) <10) & (self.df.class_name.isin(["truck", "bus"]))]        
        bigcar_df_frame = set(bigcar_df.json_path.to_list())
        closedvru_df = self.df[(abs(self.df.x) < 8) & (abs(self.df.y) <10) & (self.df.class_name.isin(["pedestrian", "tricycle", "rider"]))]  
        closedvru_df_frame = set(closedvru_df.json_path.to_list())
        
        res_pd = pd.DataFrame({"Amount": [len(bigcar_df_frame), len(closedvru_df_frame)],
                                "Ratio": [(len(bigcar_df_frame)/self.total_frames_number), (len(closedvru_df_frame)/self.total_frames_number)],
                                "Bbox_Amount": [len(bigcar_df), len(closedvru_df)],
                                })
        res_pd.index = ["bigcar", "closedvru"]
        print(res_pd)
        save_path = "%s/scenario_stats_result.xlsx" % self.save_dir
        res_pd.to_excel(save_path, index=True, header=True)
        self.logger.debug("Scenario Stats Saved in %s" % save_path)


    def build_df(self, ) -> pd.DataFrame:
        dm = data_manager_creator(self.cfg)
        dm.load_from_json()
        dm.save_to_pickle(self.cfg.FINAL_DATAFRAME_PATH)
        
        return dm.df
    
    
    def card_generator_json(self, project: str, media_name: str, dir: list):
        card_inst = CardOperation()
        card_id = card_inst.create_card_w_append(project=project, media_name=media_name, target_dir=dir) # 设置生成卡片的 project 和 media_name
        res = {"project": project, "id": card_id, "name": media_name}
        self.logger.info(res)
        return res
    
    
    def diagnose(self, ):
        self.city_stats()
        self.time_stats()
        self.info_stats()
        self.scenario_bbox_stats()
        self.city_heat_distribution()  

    
if __name__ == '__main__':
    class StatisticsManagerConfig:
        NAME = "sidecam_ori"
        DATAFRAME_PATH = "/root/data_hospital_data/0728v60/%s/dataframes/logical_dataframe.pkl" % NAME
        SAVE_DIR = "/root/data_hospital_data/0728v60/%s/statistics_manager" % NAME
        
    statistics_manager = StatisticsManager(StatisticsManagerConfig)
    statistics_manager.diagnose()
