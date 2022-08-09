import matplotlib.pyplot as plt

from src.data_manager.data_manager import load_from_pickle

plt.rc('font', size=16)
plt.rc('axes', titlesize=24) 
plt.rc('axes', labelsize=20)
from datetime import time
import pandas as pd


TYPE_MAP = {'car': 'car', 'van': 'car', 
'truck': 'truck', 'forklift': 'truck',
'bus':'bus', 
'rider':'rider',
'rider-bicycle': 'rider', 'rider-motorcycle':'rider', 
'rider_bicycle': 'rider', 'rider_motorcycle':'rider',
'bicycle': 'bicycle', 'motorcycle': 'bicycle',
'tricycle': 'tricycle', 'closed-tricycle':'tricycle', 'open-tricycle': 'tricycle', 
'closed_tricycle':'tricycle', 'open_tricycle': 'tricycle', 'pedestrian': 'pedestrian',
}


class StatsDoctor():
    def __init__(self, cfg: dict) -> None:
        self.df = load_from_pickle(cfg.FINAL_DATAFRAME_PATH)
        self.save_dir = cfg.SAVE_DIR
        
    
    def addlabels(self, x, y):
        for i in range(len(x)):
            plt.text(i, y[i], y[i], ha = 'center')
        
        
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
        print(df_concat)
        df_concat.to_excel(save_path, index=True,header=False)
        
        self.logger.debug("City Stats Saved in %s" % save_path)
        
    
    def time_stats(self,):
        
        time_dict = dict()
        time_ratio_dict = dict()
        night_df = self.df[(self.df["time"] < time(6, 0, 0)) | (self.df["time"] > time(19, 0, 0))]
        night_json = list(set(night_df.json_path.to_list()))
        time_dict["night"] = len(night_json)
        time_dict["day"] = len(self.total_frames) - time_dict["night"]
        self.output_result_txt(night_json, "night")
        
        total_values = sum([val for val in time_dict.values()])
        time_ratio_dict["night"] = time_dict["night"]/total_values
        time_ratio_dict["day"] = time_dict["day"]/total_values
        time_ratio_dict["empty"] = time_dict["empty"]/total_values

        
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
        print(df_concat)
        df_concat.to_excel(save_path, index=True,header=False)
        self.logger.debug("Time Stats Saved in %s" % save_path)
    
    
    def info_stats(self,):
        self.df['class_name_map'] = self.df['class_name'].map(TYPE_MAP)
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

            df = pd.concat([self.df[info].value_counts(),self.df[info].value_counts(normalize=True)],axis=1)
            print(df)
            save_path = "%s/%s_stats_result.xlsx" % (self.save_dir, info)
            df.to_excel(save_path, index=True,header=False)
            
            self.logger.debug("%s Stats Saved in %s" % (info.capitalize(), save_path))