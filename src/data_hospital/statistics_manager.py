import os

import folium
import matplotlib.pyplot as plt
from folium.plugins import HeatMap, MarkerCluster
from configs.config_data_hospital import DataHospitalConfig, DuplicatedCheckerConfig, LogicalCheckerConfig, Config, OutputConfig
from src.data_manager.data_manager import load_from_pickle
from src.data_manager.data_manager_creator import data_manager_creator
from src.utils.logger import get_logger
import sys
import numpy as np
import json
sys.path.append("../haomo_ai_framework")
from haomoai.cards import CardOperation
from tqdm import tqdm

plt.rc('font', size=16)
plt.rc('axes', titlesize=24) 
plt.rc('axes', labelsize=20)
from datetime import time
from datetime import datetime
import pandas as pd


class StatisticsManager():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.logger = get_logger()
        
        try:
            self.df = load_from_pickle(self.cfg.DATAFRAME_PATH)
        except FileNotFoundError:
            self.logger.error("Please Check If the File Exists")
            sys.exit()     
        
        self.total_frames_number = len(set(list(self.df.json_path)))
        self.df['class_name_map'] = self.df['class_name'].map(Config.TYPE_MAP)
        
        self.save_dir = self.cfg.SAVE_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        
        
        self.card_dir = OutputConfig.OUTPUT_CARD_DIR
        os.makedirs(self.card_dir, exist_ok=True)
        

    def addlabels(self, x, y):
        for i in range(len(x)):
            plt.text(i, y[i], y[i], ha = 'center')
        
    
    def set_loader(self, load_path: str) -> set:
        if os.path.exists(load_path):
            result = set([_.strip() for _ in open(load_path)])
        else:
            result = set()
            
        return result
        
    
    def add_noinfo_data(self, error_name: str, error_lst: list, json_lst: set):
        new_df = pd.DataFrame({"json_path": list(json_lst),
                                error_name: error_lst})
        self.df = self.df.append(new_df)
    
    
    def checking_stats(self,):
        self.logger.critical("Data Checking Error Stats Summary")
        self.dup_jsons = self.set_loader("%s/duplicated_jsons.txt" % DuplicatedCheckerConfig.SAVE_DIR)
        self.dup_imgs = self.set_loader("%s/duplicated_imgs.txt" % DuplicatedCheckerConfig.SAVE_DIR)
        ori_empty = self.set_loader("%s/empty.txt" % LogicalCheckerConfig.SAVE_DIR)
        checked_jsons = set(list(self.df.json_path))
        self.real_empty_frame = ori_empty - checked_jsons
        
        self.problem_frame, self.problem_bbox = set(), set()
        self.total_frame = checked_jsons | self.dup_jsons | self.dup_imgs | ori_empty
        total_frame_amount = len(self.total_frame) + len(self.dup_jsons)
        self.problem_frame = self.dup_jsons | self.dup_imgs | self.real_empty_frame
        
        self.add_noinfo_data("dup_json", [1 for _ in range(len(self.dup_jsons))], self.dup_jsons)
        self.add_noinfo_data("dup_img", [1 for _ in range(len(self.dup_imgs))], self.dup_imgs)
        self.add_noinfo_data("empty", [1 for _ in range(len(self.real_empty_frame))], self.real_empty_frame)
        
        res_pd = pd.DataFrame()
        res_pd.set_axis(DataHospitalConfig.TOTAL_ERROR_LIST, inplace=True)
        
        total_instance_amount = len(self.df)
        for error in DataHospitalConfig.TOTAL_ERROR_LIST:
            error_df = self.df[~self.df[error].isin([0, None, np.NaN])]
            error_instance_amount = len(error_df)
            error_frame = set(list(error_df.json_path))
            self.problem_frame |= error_frame
            error_frame_amount = len(error_frame)
            res_pd.at[error, "Instance_Amount"], res_pd.at[error, "Instance_Ratio"] = error_instance_amount, error_instance_amount / total_instance_amount
            res_pd.at[error, "Frame_Amount"], res_pd.at[error, "Frame_Ratio"] = error_frame_amount, error_frame_amount / total_frame_amount 
        
        res_pd.at["total_error", "Frame_Amount"], res_pd.at["total_error", "Frame_Ratio"] = len(self.problem_frame), len(self.problem_frame) / total_frame_amount
        save_path = "%s/error_stats.xlsx" % self.save_dir
        res_pd.to_excel(save_path, index=True, header=True)
        print(res_pd)
        
        self.real_clean_frame = self.total_frame - self.real_empty_frame - self.problem_frame
        self.logger.debug("Error Stats Saved in: %s" % save_path)
    
    
    def json_outputer(self, file_name: str, json_paths: list):
        save_dir = "%s/%s" %  (self.card_dir, file_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = "%s/%s.txt" % (save_dir, file_name)
        with open(save_path, "w") as output_file:
            for json_path in tqdm(json_paths, desc= "%s Saving" % file_name):
                output_file.writelines(json_path + "\n")
        self.logger.debug("%s Has Been Saved in: %s" % (file_name, save_path))
            
    
    def error_outputer(self, ):
        error_df = self.df[["json_path"] + DataHospitalConfig.TOTAL_ERROR_LIST].fillna(0)
        frame_error_df = error_df.groupby(["json_path"]).max()

        error_json = {"date_time": str(datetime.now()),
                        "data": []}
        
        for row in tqdm(frame_error_df.itertuples(), desc="Check Resultl Output Saving", total=len(frame_error_df)):
            index = row.Index
            error_attribute = []
            for i in range(len(frame_error_df.columns)):
                # i + 1 to ignore index
                if row[i + 1] not in [0, None, np.NaN]:
                    error_attribute.append(frame_error_df.columns[i])
            
            error_dict = dict()
            error_dict["data_oss_path"] = index
            error_dict["check_result"] = error_attribute
            error_json["data"].append(error_dict)

        save_dir = "%s/check_result"  % self.card_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = "%s/check_result.json" % save_dir
        with open(save_path, "w") as output_file:
            json.dump(error_json, output_file)
        
        self.logger.debug("Check Result Has Been Saved in: %s" % save_path)
        
    
    def output_wrapper(self, ): 
        self.json_outputer("problem", self.problem_frame)
        self.json_outputer("real_empty", self.real_empty_frame)
        self.json_outputer("real_clean", self.real_clean_frame)
        self.error_outputer()

    
    def city_stats(self,):
        self.logger.critical("Data City Stats Summary")
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
        df_concat = pd.concat([df, df_ratio],axis=1)
        df_concat.set_axis(["Amount", "Ratio"], axis='columns', inplace=True)
        print(df_concat)
        df_concat.to_excel(save_path, index=True, header=True)
        
        self.logger.debug("City Stats Saved in %s" % save_path)
        
        
    def city_heat_distribution(self, ) -> None:
        self.logger.critical("Data City Heat Stats")
        m = folium.Map([39.904989, 116.405285], tiles='stamentoner', zoom_start=10)
        df = self.df.drop_duplicates(subset="json_path", keep='first')
        df = df[df["lon"].notna()]
        
        lons = df.lon.to_list()
        lats = df.lat.to_list()
        
        comb = [[lats[i], lons[i], 1] for i in range(len(lats))]
        HeatMap(comb).add_to(m)
        
        save_path = "%s/city_heatmap.html" % self.save_dir
        m.save(save_path)
        self.logger.debug("City Heat Map Saved in: %s" % save_path)
    
    
    def time_stats(self,):
        self.logger.critical("Clean Data Time Stats Summary")
        time_dict = dict()
        time_ratio_dict = dict()
        night_df = self.df[(self.df["time"] < time(3, 0, 0)) | (self.df["time"] > time(21, 0, 0))]
        day_df = self.df[(self.df["time"] >= time(3, 0, 0)) & (self.df["time"] <= time(21, 0, 0))]
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
            self.logger.critical("Clean Data %s Stats Summary" % info.capitalize())
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
        self.logger.critical("Clean Data Scenario Stats Summary")
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
        self.checking_stats()
        self.city_stats()
        self.time_stats()
        self.info_stats()
        self.scenario_bbox_stats()
        self.city_heat_distribution()  
        self.output_wrapper()

    
if __name__ == '__main__':
    class StatisticsManagerConfig:
        NAME = "sidecam_ori"
        DATAFRAME_PATH = "/root/data_hospital_data/0728v60/%s/dataframes/logical_dataframe.pkl" % NAME
        SAVE_DIR = "/root/data_hospital_data/0728v60/%s/statistics_manager" % NAME
        
    statistics_manager = StatisticsManager(StatisticsManagerConfig)
    statistics_manager.diagnose()
