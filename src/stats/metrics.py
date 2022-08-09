#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   metrics.py
@Time    :   2022/03/30 14:19:26
@Author  :   chenminghua 
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import pdb
from tqdm import tqdm
from configs.config_cam3d import StatsConfig, OutputConfig
import src.data_manager.data_manager as dm
from src.utils.logger import get_logger
from src.stats.stats_common import define_category


class Metrics(object):
    def __init__(self, df: pd.DataFrame = None) -> None:
        self.df = df
        self.logger = get_logger()
        self.metrics = None
        self.save_dir = Path(OutputConfig.OTHERS_OUTPUT_DIR)

    def load_data(self, df: pd.DataFrame = None, df_path: str = None):
        """
        load dataframe
        """
        
        self.df = df
        if df_path != None:
            self.df = dm.load_from_pickle(df_path)
        # pdb.set_trace()

    def _define_priority(self, df):
        for row in df.itertuples():
            if row.bev_case_flag != '0' and  row.bev_case_flag != '1':
                df.at[row.Index, 'priority'] = 'P1'
                continue
            if pd.isna(row.peer_id):
                if abs(row.x) > 20 or abs(row.y) > 20:
                    df.at[row.Index, 'priority'] = 'P1'
                    continue
                else:
                    df.at[row.Index, 'priority'] = 'P0' 
                    continue 
            if (abs(row.x) > 20 or abs(row.y) > 20) and (abs(df.loc[row.peer_id].x) > 20 or abs(df.loc[row.peer_id].y) > 20):
                continue
            df.at[row.Index, 'priority'] = 'P0'

    def compute_metrics(self, ):
        """
        Entrance of evaluation
        """
        self.logger.info("start compute metrics")
        # columns = ['precision_2d', 'recall_2d', 'f1_score_2d', 'tp_bev', 'fp_bev', 'fn_bev', 'precision_bev',
        # 'recall_bev', 'f1_score_bev', 'position_error', 'position_error_2sigma', 'position_error<10%',
        # 'pos_x_error','pos_x_error_2sigma', 'pos_y_error', 'pos_y_error_2sigma', 'yaw_error', 'yaw_error_2sigma', 
        # 'yaw_error<10','length_error', 'width_error']
        columns = ['precision_2d', 'recall_2d']
        rows = ['all', 'vehicle', 'vru'] + StatsConfig.ALL_CATEGORY
        self.metrics = {row: {column: 0 for column in columns} for row in rows}
        self.df = define_category(self.df)
        # self._define_priority(self.df)
        # df_p0 = self.df[((self.df.bev_case_flag == '0') | (self.df.bev_case_flag == '1')) & (self.df.priority == 'P0')]
        df_p0 = self.df[(self.df.priority == 'P0')]
        for row, metric in self.metrics.items():
            if row == 'all':
                df_select = df_p0[(df_p0['broad_category'] == 'vehicle') | (df_p0['broad_category'] == 'vru')] 
            else:
                df_select = df_p0[df_p0['broad_category'] == row]
                if row in StatsConfig.ALL_CATEGORY:
                    df_select = df_p0[df_p0['class_name'] == row]
            tp_bev, fp_bev, fn_bev = sum(df_select['flag'] == 'good'),\
                                    sum(df_select['flag'] == 'false'), \
                                    sum(df_select['flag'] == 'miss')
            tp_2d, fp_2d, fn_2d = sum((df_select['case_flag']  == '0')),\
                                    sum((df_select['case_flag'] == '2') | (df_select['case_flag'] == '22') | (df_select['case_flag'] == '21')),\
                                    sum((df_select['case_flag'] == '1') | (df_select['case_flag'] == '22') | (df_select['case_flag'] == '21'))

            self.metrics[row]['tp_2d'] = tp_2d
            self.metrics[row]['fp_2d'] = fp_2d
            self.metrics[row]['fn_2d'] = fn_2d

            self.metrics[row]['precision_2d'] = tp_2d / (tp_2d + fp_2d) if tp_2d + fp_2d != 0 else -1
            self.metrics[row]['recall_2d'] = tp_2d / (tp_2d + fn_2d) if tp_2d + fn_2d != 0 else -1
            self.metrics[row]['f1_score_2d'] = 2/(1/self.metrics[row]['precision_2d']+ 1/self.metrics[row]['recall_2d']) if \
                                        self.metrics[row]['precision_2d']!= 0 and self.metrics[row]['recall_2d'] != 0 else -1
            self.metrics[row]['tp_bev'] = tp_bev
            self.metrics[row]['fp_bev'] = fp_bev
            self.metrics[row]['fn_bev'] = fn_bev

            self.metrics[row]['precision_bev'] = tp_bev / (tp_bev + fp_bev) if tp_bev + fp_bev != 0 else -1
            self.metrics[row]['recall_bev'] = tp_bev / (tp_bev + fn_bev) if tp_bev + fn_bev != 0 else -1
            self.metrics[row]['f1_score_bev'] = 2/(1/self.metrics[row]['precision_bev']+ 1/self.metrics[row]['recall_bev']) if \
                                        self.metrics[row]['precision_bev']!= 0 and self.metrics[row]['recall_bev'] != 0 else -1
            
            self.metrics[row]['position_error'] = df_select[df_select['case_flag']=='0'].dis_ratio.mean()
            self.metrics[row]['pos_error_2sigma'] = np.nanquantile(df_select[df_select['case_flag']=='0'].dis_ratio, StatsConfig.SIGMA2)
            # self.metrics[row]['pos_error_2sigma'] = np.nanquantile(df_select[df_select['case_flag']=='0'].dis_ratio, StatsConfig.SIGMA2)

            self.metrics[row]['yaw_error'] = df_select[df_select['case_flag']=='0'].yaw_diff.mean() / 3.14*180
            self.metrics[row]['yaw_error_2sigma'] = np.nanquantile(df_select[df_select['case_flag']=='0'].yaw_diff, StatsConfig.SIGMA2)/3.14*180
            self.metrics[row]['yaw_error<10'] = sum(df_select[df_select['case_flag']=='0'].yaw_diff < 10/180*3.14)  / sum(df_select['case_flag']=='0') \
                                                if sum(df_select['case_flag']=='0') != 0 else -1

        return self.metrics
        

    def show_metrics(self, ):
        index = list(self.metrics.keys())
        columns = list(self.metrics[index[0]].keys())
        m = {'type' : index}
        for column in columns:
            m[column] = [self.metrics[p][column] for p in index]
        df_metrics = pd.DataFrame(m)
        save_path = "%s/metrics.xlsx" % self.save_dir
        df_metrics.to_excel(save_path)
        self.logger.debug("Metrics Has Been Saved in: %s" % save_path)
        print(df_metrics)
        self.logger.info('\t'+ df_metrics.to_string().replace('\n', '\n\t') + '\n')