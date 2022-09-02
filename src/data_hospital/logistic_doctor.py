import shutil
import json
import os
import pickle
import math

import pandas as pd
import tqdm
from configs.config import (OutputConfig, ReprojectDoctorConfig, VisualizationConfig, CoorTransConfig, LogisticDoctorConfig)
from src.data_hospital.coor_trans_doctor import CoorTransDoctor
from src.data_manager.data_manager import DataManager, load_from_pickle
from src.data_manager.data_manager_creator import data_manager_creator
from src.utils.logger import get_logger
from src.utils.struct import parse_obs
from src.visualization.visualization import Visualizer


# WEY MOKA
self_width = 1.96
self_length = 4.875


class LogisticDoctor(DataManager):
    def __init__(self, cfg: dict) -> None:
        self.logger = get_logger()
                
        dm = data_manager_creator(cfg)
        dm.load_from_json()
        dm.save_to_pickle(cfg.LOGISTIC_DATAFRAME_PATH)
        self.df = dm.df
        
        with open(cfg.JSON_PATH) as total_file:
            self.total_frames = set([_.strip() for _ in total_file])

        self.logger.info("DataFrame Loaded")
            
        self.instance_number = len(self.df)
        self.prob_frames = set()
        self.clean_instances = set()
        self.frames = set(self.df.json_path.to_list())
        self.noemp_frame_number = len(self.frames)
        self.total_frame_number = len(self.total_frames)
        self.ver = Visualizer(VisualizationConfig)
        self.txt_output_dir = cfg.SAVE_DIR
        self.json_output_dir = OutputConfig.OUTPUT_DIR
        self.res_dict = dict()
        self.online = cfg.ONLINE
        self.empty_frames = []
        self.error_list = ["bbox_error", "coor_error"]
        self.vis = cfg.VIS
        self.coor = cfg.COOR
        
    def instance_erros(self, ) -> None:
        """
        Summary
        -------
            Instance Level Error Stats
        
        """
        
        self.logger.critical("Instance Level Errors Diagnosing")
        instance_dict = dict()
        instance_per_dcit = dict()
        
        for error in self.error_list:
            
            clean_indexes = set(self.df[self.df[error] == 0].index.to_list())
            self.clean_instances = self.clean_instances.intersection(clean_indexes) if len(self.clean_instances) != 0 else clean_indexes
                
            groupby_error = self.df.groupby(error).size()
            instance_dict["%s_sum" % error] = sum(groupby_error[1:])
            instance_per_dcit["%s_sum" % error] = round(instance_dict["%s_sum" % error] / self.instance_number, 4)
            
            for ind, error_amount in enumerate(groupby_error):
                instance_dict["%s_%d" % (error, ind)] = error_amount
                instance_per_dcit["%s_%d" % (error, ind)] = round(error_amount / self.instance_number, 4)

        instance_dict["error_sum"] = self.instance_number - len(self.clean_instances)
        instance_per_dcit["error_sum"] = round(instance_dict["error_sum"] / self.instance_number, 4)

        self.res_dict["instance"] = instance_dict
        self.res_dict["instance_per"] = instance_per_dcit
                
                
    def frame_errors(self, ) -> None:
        """
        Summary
        -------
            Frame Level Error, output TXT of json_url in diffent file
        
        """
        
        self.logger.critical("Frame Level Errors Cleaning")
        frame_dict = dict()
        frame_per_dict = dict()
        
        for error in self.error_list:
            
            gb_df = self.df.groupby(["json_path", error]).size().unstack(fill_value=0)
            gb_df["sum"] = gb_df.sum(axis=1)
            
            error_df = gb_df[gb_df["sum"] != gb_df[0]] if 0 in gb_df else gb_df
            frame_dict["%s_sum" % error] = len(error_df)
            frame_per_dict["%s_sum" % error] = round(frame_dict["%s_sum" % error] / self.noemp_frame_number, 4)
            
            self.prob_frames |= set(error_df.index.to_list())
            self.output_result_txt(error_df.index.to_list(), error)
            
            correct_df = gb_df[gb_df["sum"] == gb_df[0]] if 0 in gb_df else pd.DataFrame()
            frame_dict["%s_%d" %(error, 0)] = len(correct_df)
            frame_per_dict["%s_%d" %(error, 0)] = round(frame_dict["%s_%d" %(error, 0)] / self.noemp_frame_number, 4)
            
            for flag in gb_df.columns.to_list()[1:-1]:
                flag_error_df = gb_df[gb_df[flag] != 0]
                
                frame_dict["%s_%d" %(error, flag)] = len(flag_error_df)
                frame_per_dict["%s_%d" %(error, flag)] = round(frame_dict["%s_%d" %(error, flag)] / self.noemp_frame_number, 4)

        self.clean_frames = self.frames - self.prob_frames
        self.empty_frames = self.total_frames - self.prob_frames - self.clean_frames
        clean_empty = self.clean_frames | self.empty_frames

        frame_dict["error_sum"] = len(self.prob_frames)
        frame_per_dict["error_sum"] = round(frame_dict["error_sum"] / self.total_frame_number, 4)
        frame_dict["total"] = self.total_frame_number
        frame_per_dict["total"] = round(frame_dict["total"] / self.total_frame_number, 4)
        frame_dict["clean"] = len(self.clean_frames)
        frame_per_dict["clean"] = round(frame_dict["clean"] / self.total_frame_number, 4)
        frame_dict["empty"] = len(self.empty_frames)
        frame_per_dict["empty"] = round(frame_dict["empty"] / self.total_frame_number, 4)
        frame_dict["clean_empty"] = len(clean_empty)
        frame_per_dict["clean_empty"] = round(frame_dict["clean_empty"] / self.total_frame_number, 4)
        
        self.output_result_txt(self.clean_frames, "clean")
        self.output_result_txt(self.empty_frames, "empty")
        self.output_result_txt(clean_empty, "clean_empty")
        
        self.res_dict["frame"] = frame_dict
        self.res_dict["frame_per"] = frame_per_dict
        
        df = pd.DataFrame.from_dict(self.res_dict).T
        save_path = "%s/error_stats_result.xlsx" % self.txt_output_dir
        df.round(4)
        df.to_excel(save_path)
        print(df)
        self.logger.debug("Error Stats Saved in %s" % save_path)
        
        
    def card_errors(self, ) -> None:
        gb_df = self.df.groupby(["card_id", "has_error"]).size().unstack(fill_value=0)
        gb_df["error_rate"] = gb_df[1] / gb_df[1] + gb_df[0]
        print(gb_df)
    
    
    def output_result_txt(self, json_paths: list, file_name: str) -> None:
        """
        Summary
        -------
            Output json_url to txt file, ready to generate new card

        Parameters
        ----------
            json_paths: list
                list of json_path to be outpu
            file_name: str
                file name
        """
        
        if self.online:
            parent_path = "%s/%s" % (self.json_output_dir, file_name)
            os.makedirs(parent_path, exist_ok=True)
            for ind, json_path in enumerate(json_paths):
                with open(json_path) as json_file:
                    json_obj = json.load(json_file)
                    with open("%s/%d.json" % (parent_path, ind), 'w') as output_file:
                        json.dump(json_obj, output_file)
            
        else:
            os.makedirs(self.txt_output_dir, exist_ok=True)
            with open("%s/%s.txt" % (self.txt_output_dir, file_name), "w") as output_file:
                for json_path in json_paths:
                    output_file.writelines(json_path + "\n")

                
    def txt_for_reproejct(self, ) -> None:
        df_reproject = load_from_pickle(ReprojectDoctorConfig.REPROJECT_DATAFRAME_PATH)
        carday_ids = set(df_reproject.carday_id)
        self.logger.debug("Reproject Ready File Has Been Saved in %s" % ReprojectDoctorConfig.LOAD_PATH)
        for carday_id in tqdm.tqdm(carday_ids, desc="Spliting Json Paths Based on CarDay"):
            json_paths = list(set(df_reproject[df_reproject['carday_id'] == carday_id].json_path.tolist()))
            shutil.os.makedirs(ReprojectDoctorConfig.LOAD_PATH, exist_ok=True)
            with open("%s/%s.txt" % (ReprojectDoctorConfig.LOAD_PATH, carday_id), "w") as output_file:
                for json_path in json_paths:
                    output_file.writelines(json_path + "\n")
        
                
    def visualization(self, sample_no: int) -> None:
        """
        Summary
        -------
            Visualization BadCase for differnet type
        
        """
        
        self.logger.debug("Example Error Images Saved in: %s" % self.ver.save_dir)
        coor_trans = CoorTransDoctor(CoorTransConfig)
        for error_type in self.error_list:
            for flag in self.df[error_type].unique().tolist():
                # if flag != 0:
                error_df = self.df[self.df[error_type] == flag]
                error_df = error_df.sample(n=min(sample_no, len(error_df)))
                
                if self.coor == "Lidar":
                    objs = []
                    for _ in range(len(error_df)):
                        json_map = coor_trans.get_ann_info(error_df.iloc[_].json_path)
                        id = error_df.iloc[_].id
                        new = error_df.iloc[_]
                        
                        for object in json_map['objects']:
                            if object["id"] == id:
                                try:
                                    attr = object['3D_attributes']['position']
                                    new.x = attr["x"]
                                    new.y = attr["y"]
                                    new.z = attr["z"]
                                    new.yaw = object['3D_attributes']['rotation']["yaw"]
                                except:
                                    pass
                                    
                        objs.append(parse_obs(new))
                
                else:
                    objs = [parse_obs(error_df.iloc[_]) for _ in range(len(error_df))]
                    
                for ind, obj in tqdm.tqdm(enumerate(objs), desc="Saving %s_%d Images" % (error_type, flag)):
                    self.ver.draw_bbox_0([obj], "%s/%s_%d/%d.jpg" % (self.ver.save_dir, error_type, flag, ind))
                    self.ver.plot_bird_view([obj], "%s/%s_%d/%d_bev.jpg" % (self.ver.save_dir, error_type, flag, ind))
                    with open("%s/%s_%d/%d.txt" % (self.ver.save_dir, error_type, flag, ind), "w") as output_file:
                        info = "%s\n%s\n%f %f %f\n%s\ntruncation=%s\n%s" % (str(obj.bbox), obj.camera_orientation, obj.x, obj.y, obj.z, str(obj.flag), str(obj.truncation), str(obj.json_path))
                        output_file.writelines(info)
                            
    
    def construct_reproject_dataframe(self, ) -> None:
        reporject_df = self.df[~self.df.json_path.isin(self.df[self.df.has_error == 1].json_path)]
        pickle_obj = {"info": {}, "df": reporject_df}
        with open(ReprojectDoctorConfig.REPROJECT_DATAFRAME_PATH, "wb") as pickle_file: 
            pickle.dump(pickle_obj, pickle_file)
        
        self.logger.critical("DataFrame Saved: %s" % ReprojectDoctorConfig.REPROJECT_DATAFRAME_PATH)
        
                
    def diagnose(self,):
        self.instance_erros()
        self.frame_errors()
        self.construct_reproject_dataframe()
        
        if self.vis:
            self.visualization(100)
            
    
    def coor_checker(info:dict) -> None:
        if LogisticDoctorConfig.COOR == "Car":
            LogisticDoctor.coor_checker_car(info)
        else:
            LogisticDoctor.coor_checker_lidar(info)
        
        
    def coor_checker_car(info: dict) -> None:
        """
        Summary
        -------
            Check if the coordinate system trans has error, and assign the corresponding flag for the coor_error
                1: trans error
                
        Parameters
        ----------
            info: dict
                the info json object
                
        """
        
        ori = info["camera_orientation"]
        x, y, z, h = info["x"], info["y"], info["z"], info["height"]
        
        if (info["truncation"] in [1, 2]) and (abs(info["x"]) < 50) and (abs(info["y"]) < 50):
            if z - h / 2 < -1:
                if abs(x) < 10 and abs(y) < 10:
                    info["coor_error"] = 3
                else:
                    info["coor_error"] = 2
            
        else:
            x = x - 2 * self_length / 3

            # tolerance for side cam
            side_co_x = math.tan(10 * math.pi/180)
            side_cox = abs(x) * side_co_x
            
            side_co_y = math.tan(10 * math.pi/180)

            # tolerance for front & rear cam
            front_co = math.tan(60* math.pi/180)

            if ori == "front_left_camera":
                y = y - self_width / 2
                side_coy = abs(y) * side_co_y
                if x < -side_coy or y < -side_cox:
                    info["coor_error"] = 1
                    
            elif ori == "front_right_camera":
                y = y + self_width / 2
                side_coy = abs(y) * side_co_y
                if x < -side_coy or y > side_cox:
                    info["coor_error"] = 1
                    
            elif ori == "rear_left_camera": 
                y = y - self_width / 2
                side_coy = abs(y) * side_co_y
                if x > side_coy or y < -side_cox:
                    info["coor_error"] = 1
                    
            elif ori == "rear_right_camera":
                y = y + self_width / 2
                side_coy = abs(y) * side_co_y
                if x > side_coy or y > side_cox:
                    info["coor_error"] = 1
                    
            elif ori == "front_middle_camera": 
                front_side_cox = abs(x) * front_co
                if x < 0 or y < - front_side_cox or y > front_side_cox:
                    info["coor_error"] = 1
                    
            elif ori == "rear_middle_camera": 
                x = x + 2 * self_length / 3
                front_side_cox = abs(x) * front_co
                if x > 0 or y < - front_side_cox or y > front_side_cox: 
                    info["coor_error"] = 1
                    
    
    def coor_checker_lidar(info: dict) -> None:
        """
        Summary
        -------
            Check if the coordinate system trans has error, and assign the corresponding flag for the coor_error
                1: trans error
                
        Parameters
        ----------
            info: dict
                the info json object
                
        """
        
        ori = info["camera_orientation"]
        x, y, z, h = info["x"], info["y"], info["z"], info["height"]
        
        if (info["truncation"] in [1, 2]) and (abs(info["x"]) < 50) and (abs(info["y"]) < 50):
            if z - h / 2 > -1:
                if abs(x) < 10 and abs(y) < 10:
                    info["coor_error"] = 3
                else:
                    info["coor_error"] = 2
        
        else:
            y = y + self_length / 6

            # tolerance for side cam
            side_co_y = math.tan(10 * math.pi/180)
            side_coy = abs(y) * side_co_y
            side_co_x =  math.tan(10 * math.pi/180)

            # tolerance for front & rear cam
            front_co = math.tan(60 * math.pi/180)

            if ori == "front_left_camera":
                x = x - self_width / 2
                side_cox = abs(x) * side_co_x
                if x < -side_coy or y > side_cox:
                    info["coor_error"] = 1
                    
            if ori == "front_right_camera":
                x = x + self_width / 2
                side_cox = abs(x) * side_co_x
                if x > side_coy or y > side_cox:
                    info["coor_error"] = 1
                    
            if ori == "rear_left_camera":
                x = x - self_width / 2
                side_cox = abs(x) * side_co_x
                if x < -side_coy or y < -side_cox:
                    info["coor_error"] = 1
                    
            if ori == "rear_right_camera":
                x = x + self_width / 2 
                side_cox = abs(x) * side_co_x
                if x > side_coy or y < -side_cox:
                    info["coor_error"] = 1
                    
            if ori == "front_middle_camera":
                front_side_coy = abs(y) * front_co
                if y > 0 or x < - front_side_coy or x > front_side_coy:
                    info["coor_error"] = 1
                    
            if ori == "rear_middle_camera":
                y = y - self_length / 6 - self_length / 2
                front_side_coy = abs(y) * front_co
                if y < 0 or x < - front_side_coy or x > front_side_coy:
                    info["coor_error"] = 1
            
    
    def bbox_checker(info: dict, width: int = None, height: int = None) -> None:
        """
        Summary
        -------
            Check if the bbox has error, and assign the corresponding flag for the bbox_error
                1: x/y/w/h out of bounds
                2: x+w/y+h out of bounds

        Parameters
        ----------
            info: dict
                the info json object
                
        """
        width = info["img_width"] if width is None else width
        height = info["img_height"] if height is None else height
        
        if (info["bbox_x"] > (width * 1.02)) \
        or (info["bbox_y"] > (height * 1.02)) \
        or (info["bbox_x"] < - (width * 0.02)) \
        or (info["bbox_y"] < -(height * 0.02)) \
        or (info["bbox_w"] < 0) \
        or (info["bbox_h"] < 0) \
        or (info["bbox_w"] > width * 1.02) \
        or (info["bbox_h"] > height * 1.02):
            info["bbox_error"] = 1
        
        elif (info["bbox_x"] + info["bbox_w"] > (width * 1.02)) \
        or (info["bbox_y"] + info["bbox_h"] > (height * 1.02)):
            info["bbox_error"] = 2
            
            
if __name__ == '__main__':
    logistic_doctor = LogisticDoctor(LogisticDoctorConfig)
    logistic_doctor.df = pd.read_pickle("/root/data_hospital_data/0728v60/v31_0823/dataframes/reproject_dataframe.pkl")["df"]
    if logistic_doctor.vis:
        logistic_doctor.visualization(100)
        