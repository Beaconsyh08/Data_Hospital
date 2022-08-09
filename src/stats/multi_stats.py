#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   error_analysis.py
@Time    :   2022/03/30 14:20:10
@Author  :   chenminghua 
'''
import collections
import math
import os
import threading
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import src.data_manager.data_manager as dm
from configs.config import (Config, OutputConfig, StatsConfig,
                            VisualizationConfig)
from src.utils.common import compute_iou
from src.utils.file_io import write_json
from src.utils.logger import get_logger
from src.utils.struct import parse_obs
from src.visualization.visualization import Visualizer
from tqdm import tqdm


class MultiStats(object):
    def __init__(self, df_multi_version: List[pd.DataFrame] = None) -> None:
        if df_multi_version is None:
            df_multi_version = []
        self.df_multi_version = df_multi_version
        self.logger = get_logger()
        self.save_dir = Path(OutputConfig.OTHERS_OUTPUT_DIR)
        self.visualizer = Visualizer(VisualizationConfig)

    def load_data(self, df_multi_version: List[pd.DataFrame] = None):
        """
        load dataframe
        """
        if df_multi_version is None:
            df_multi_version = []
        self.df_multi_version = df_multi_version
        self.qa_base_df = self.df_multi_version[0]
        self.df = dm.load_from_pickle(StatsConfig.DATAFRAME_PATH)
        # pdb.set_trace()
    
    # def process(self, show=True):
    #     self.logger.info("Start Multi Version Analysis")
    #     if len(self.df_multi_version) < 2:
    #         self.logger.warning("There are less than two multi version files, multi version comparison is not allowed !")
    #         return
    #     self.qa_base_df = self.df_multi_version[0]
    #     df_base = self.df_multi_version[1]
    #     dic_multi = collections.OrderedDict(Index=[], flag_base=[])
    #     dt_base = {}
    #     dt_new = {}
    #     for row in tqdm(df_base.itertuples(), total=len(df_base)):
    #         if row.dtgt != 'dt' or row.flag != 'false' or row.priority != 'P0':
    #             continue
    #         bbox = [row.bbox_x,row.bbox_y,row.bbox_w,row.bbox_h]
    #         img_name = row.Index.split('@')[0]
    #         if img_name not in dt_base:
    #             dt_base[img_name] = {}
    #         dt_base[img_name][row.Index] = bbox
    #     for row in tqdm(self.df_multi_version[-1].itertuples(), total=len(self.df_multi_version[-1])):
    #         if row.dtgt != 'dt' or row.flag != 'false' or row.priority != 'P0':
    #             continue
    #         bbox = [row.bbox_x,row.bbox_y,row.bbox_w,row.bbox_h]
    #         img_name = row.Index.split('@')[0]
    #         if img_name not in dt_new:
    #             dt_new[img_name] = {}
    #         dt_new[img_name][row.Index] = bbox
    #     hard_case_index = []
    #     regress_case_index = []
    #     for img_name, dt in dt_new.items():
    #         for new_id, new_bbox in dt.items():
    #             flag = False
    #             if img_name  in dt_base:
    #                 for base_id, base_bbox in dt_base[img_name].items():
    #                     if compute_iou(new_bbox, base_bbox) > 0.5:
    #                         hard_case_index.append(new_id)
    #                         flag = True
    #             if not flag:
    #                 regress_case_index.append(new_id)
    #     for index in hard_case_index:
    #         row = self.df_multi_version[-1].loc[index]
    #         try:
    #             obs_list = [parse_obs(row)]
    #             peer_id = row.peer_id
    #         except:
    #             pass
    #         if not pd.isna(peer_id):
    #             obs_list.append(parse_obs(self.df_multi_version[-1].loc[row.peer_id]))
    #         img_name = Path(index).stem
    #         save_path = os.path.join('/cpfs/output/other/false', '_'.join([row.class_name,  str(row.cluster_id),  row.error_type, str(row.score), img_name, str(row.id)+'.jpg']))
    #         bev_save_path = str(Path(save_path).parent/(str(Path(save_path).stem)+'_bev.jpg'))

    #         t0 = threading.Thread(target=self.visualizer.draw_bbox_0, args=(obs_list, save_path))
    #         t0.start()
    #         t1 = threading.Thread(target=self.visualizer.plot_bird_view, args=(obs_list, bev_save_path))
    #         t1.start()
        
    #     return self.indicator_panel
    def get_flag_from_multi_df(self, ):
        df_base = self.df_multi_version[0]
        dic_multi = collections.OrderedDict(Index=[], flag_base=[])

        dic_others_version = [df.to_dict() for df in self.df_multi_version[1:]]
        for row in tqdm(df_base.itertuples(), total=len(df_base)):
            index,flag = getattr(row, 'Index'), getattr(row, 'flag')
            if flag not in ['good', 'miss']:
                continue
            dic_multi['Index'].append(index)
            dic_multi['flag_base'].append(flag)

            for idx, dic_version in enumerate(dic_others_version):
                column_name = 'flag_' + str(idx+1)        
                if column_name not in dic_multi:
                    dic_multi[column_name] = []
                if index not in dic_version['flag']:
                    dic_multi[column_name].append(None)
                    continue
                dic_multi[column_name].append(dic_version['flag'][index])
        df_multi_comp = pd.DataFrame.from_dict(dic_multi)
        df_multi_comp.set_index('Index',  inplace=True)

        return df_multi_comp
    
    
    def define_changes(self, save_df=True):
        self.logger.info("Start Multi Version Analysis")
        if len(self.df_multi_version) < 2:
            self.logger.warning("There are less than two multi version files, multi version comparison is not allowed !")
            return
        df_multi_comp = self.get_flag_from_multi_df()
        self._define_case_change(df_multi_comp)

        self.df['multi_version_info'] = None
        for row in tqdm(df_multi_comp.itertuples(), total=len(df_multi_comp)):
            index = row.Index
            try:
                self.df.at[index, 'multi_version_info'] = row.multi_version
                self.df.at[index, 'flag_base'] = row.flag_base
            except Exception:
                self.logger.warning("%s not in dataframe, can't save multi_version_info" %(index))
        if save_df:
            temp_dm = dm.DataManager(self.df)
            temp_dm.save_to_pickle(str(self.save_dir / 'qa_base_multi_version.pkl'))
        print(df_multi_comp)
        
        return self.df

    
    def compute_metrics(self, df):
        # priority_multi_info = self.qa_base_df.groupby(['priority', 'multi_version_info']).size().unstack(fill_value=0).to_dict('index')
        if 'multi_version_info' not in df.columns or 'flag_base' not in df.columns:
            self.logger.warning("multi_version_info or flag_base not in df, can't output regress/progress/easy metrics")
            return
        num_miss, num_false, num_good = len(df['flag_base'] == 'miss'), len(df['flag_base'] == 'false'), len(df['flag_base'] == 'good')
        num_fixed, num_hard, num_retrogression= len(df['multi_version_info'] == 'fixed'), len(df['multi_version_info'] == 'hard'),\
                                    len(df['multi_version_info'] == 'retrogression')

        self.indicator_panel = {'num_p0_miss_case': num_miss, 'num_p0_false_case': num_false, \
                                'fixed': num_fixed, 'hard': num_hard, 'retrogression': num_retrogression}

        # case_discribe_list = ['easy', 'hard', 'fixed', 'retrogression']

        self.indicator_panel['p0_fixed_rate'] = len(df['flag_base'] == 'fixed') / num_miss  if num_miss != 0 else -1
        self.indicator_panel['p0_retrogression_rate'] = len(df['flag_base'] == 'retrogression') / num_miss if num_miss != 0 else -1
        case_change_in_class = df.groupby(['class_name', 'multi_version_info']).size().unstack(fill_value=0)
        case_change_in_class["absolute_fix"] = case_change_in_class["fixed"] - case_change_in_class["retrogression"]
        case_change_in_class["absolute_fix_rate"] = case_change_in_class["absolute_fix"] / (case_change_in_class["fixed"] + case_change_in_class["hard"])
        case_change_in_class.sort_values(by="absolute_fix", inplace=True, ascending=False)
        save_path = "%s/case_change_in_class.xlsx" % self.save_dir
        case_change_in_class.to_excel(save_path)
        self.logger.info('\t'+ case_change_in_class.to_string().replace('\n', '\n\t'))
        self.logger.debug("Case Change In Class Has been Saved in: %s" % save_path)
        # print(case_change_in_class)
        return self.indicator_panel
    
    def show_multi_version_cases(self, df):
        self.logger.info("Saving fixed/retrogression/hard cases in multi_version_analysis.")
        for row in tqdm(df.itertuples(), total=len(df)):
            index, multi_version =  getattr(row, 'Index'), getattr(row, 'multi_version_info')
            if multi_version == 'easy':
                continue
            try:
                if self.qa_base_df.at[index, 'priority'] != 'P0':
                    continue
            except Exception:
                continue
            img_path = index.split('@')[0]
            img_name = Path(img_path).stem
            bbox_list = [[self.qa_base_df.at[index, 'bbox_x'], self.qa_base_df.at[index, 'bbox_y'],
                            self.qa_base_df.at[index, 'bbox_w'],self.qa_base_df.at[index, 'bbox_h']]]
            flag, class_name, id = self.qa_base_df.at[index, 'flag'],\
                                    self.qa_base_df.at[index, 'class_name'],\
                                    self.qa_base_df.at[index, 'id']
            # print(self.df_multi_version[-1].columns)  
            # cluster_id = str(self.df_multi_version[-1].at[index, 'cluster_id'])
            # error_type = str(self.df_multi_version[-1].at[index, 'error_type'])
            score = str(self.df_multi_version[-1].at[index, 'score'])

            # save_path = os.path.join('/cpfs/output/other/cluster', '_'.join([cluster_id, multi_version, error_type, score, class_name, img_name, str(id)+'.jpg']))
            # save_path = os.path.join('/cpfs/output/other/cluster', '_'.join([class_name,  cluster_id, multi_version, error_type, score, img_name, str(id)+'.jpg']))

            # class_name_list = [class_name]
            # self.visualizer.draw_bbox(img_path, bbox_list, save_path, text_list = class_name_list)
            # t = threading.Thread(target=self.visualizer.draw_bbox, args=(img_path, bbox_list, save_path, class_name_list))
            # t.start()

            try:
                obs_list = [parse_obs(self.df_multi_version[-1].loc[index])]
                peer_id = self.df_multi_version[-1].loc[index].peer_id
            except:
                pass
            if not pd.isna(peer_id):
                try:
                    obs_list.append(parse_obs(self.df_multi_version[-1].loc[peer_id]))
                except:
                    pass
            try:
                # save_path = os.path.join('/cpfs/output/other/cluster', '_'.join([class_name,  multi_version, score, img_name, str(id)+'.jpg']))
                save_path = os.path.join('/cpfs/output/other/cluster', '_'.join([class_name,  multi_version, score, img_name, str(id)+'.jpg']))
                bev_save_path = str(Path(save_path).parent/(str(Path(save_path).stem)+'_bev.jpg'))
            except:
                continue
            # print(save_path)
            # print(obs_list[0])
            t0 = threading.Thread(target=self.visualizer.draw_bbox_0, args=(obs_list, save_path))
            t0.start()
            t1 = threading.Thread(target=self.visualizer.plot_bird_view, args=(obs_list, bev_save_path))
            t1.start()
    
    def _define_case_change(self, df_multi_comp):
        # df_multi_comp
        comp_dict = df_multi_comp.T.to_dict('list')
        case_discribe = []
        for row in tqdm(df_multi_comp.itertuples(), total=len(df_multi_comp)):
            index = getattr(row, 'Index')
            values = list(row)[1:]
            if 'miss' not in values:
                case_discribe.append('easy')
            elif 'good' not in values:
                case_discribe.append('hard')
            elif values[0] == 'miss' and values[-1] == 'good':
                case_discribe.append('fixed')
            elif values[0] == 'good' and values[-1] =='miss':
                case_discribe.append('retrogression')
            else:
                case_discribe.append('fluctuate')
        df_multi_comp['multi_version'] = case_discribe
    
    def show_result(self, ):
        pass
