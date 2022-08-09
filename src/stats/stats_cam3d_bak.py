#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   stats.py
@Time    :   2021/11/15 20:17:09
@Email   :   songyuhao@haomo.ai
@Author  :   YUHAO SONG 

Copyright (c) HAOMO.AI, Inc. and its affiliates. All Rights Reserved
"""

from ast import Return
import collections
import math
import os
import random
import shutil
import threading
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.data_manager.data_manager as dm
from configs.config import Config, StatsConfig, VisualizationConfig, OutputConfig
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from src.classification.clustering import Clustering
from src.stats.base_stats import BaseStats
from src.utils.common import compute_iou
from src.utils.file_io import write_json
from tqdm import tqdm


class StatsCam3D(BaseStats):

    def __init__(self, df: pd.DataFrame = None, df_path: str = None, 
                       emb_path: str = None, config: dict = None) -> None:
        
        super(StatsCam3D, self).__init__(config=config, df=df,
                                         df_path=df_path,emb_path=emb_path)
        self.indicator_panel = {}
        self.save_dir = Path(OutputConfig.OTHERS_OUTPUT_DIR)
        self.indicator_path = self.save_dir / 'indicator_panel.json'

    def cases_overall_stats(self, dim_col: str) -> pd.Series:
        """
        Summary
        -------
            This mehtod present the results of overall cases stats based on chosen dimension, size, flag, dt/gt

        Parameters
        ----------
            dim_col: str
                the main dimension to prestne the results, typical choice is "class_name" or "clsuter_id"

        Returns
        -------
            pd.Series: a Series object contains the wanted stats
        """

        return self.df.groupby([dim_col, "flag", "dtgt"]).size()


    def get_description(self, dim_col: str) -> pd.DataFrame:
        """
        Summary
        -------
            This method presetn the common stats for the dataframe,
            could present Precision, Recall, F1-score, True Positive, False Positive, False Negative
            based on different dimension, typical choice is "class_name" of "cluster"

        Parameters
        ----------
            dim_col: str
                the dimension to present the stats, typical choice is "class_name" of "cluster_id"

        Returns
        -------
            pd.DataFrame: The wanted stats in DataFrame
        """
        
        uniques = list(self.df[dim_col].unique())
        stats = self.cases_overall_stats(dim_col)
        print(stats)
        stats_dicts = [{unique: {"good": 0, "miss": 0, "false": 0}} for _, unique in enumerate(uniques)]
        total_stats = []

        for ind, stat in enumerate(stats):
            for dic in stats_dicts:
                dic_key = list(dic.keys())[0]
                stat_index = stats.index[ind]
                if dic_key == stat_index[0]:
                    dic[dic_key][stat_index[1]] = stat

        total_tp, total_fp, total_fn = 0, 0, 0

        for each in stats_dicts:
            tp = list(each.values())[0]["good"]
            fp = list(each.values())[0]["false"]
            fn = list(each.values())[0]["miss"]
            precision = round(100 * tp / (tp + fp), 2) if tp + fp != 0 else 0
            recall = round(100 * tp / (tp + fn), 2) if tp + fn != 0 else 0
            f1_score = round(2 * precision * recall / (precision + recall), 2) if precision + recall != 0 else 0
            total_stats.append([list(each.keys())[0], tp, fp, fn, precision, recall, f1_score])
            total_tp += tp
            total_fp += fp
            total_fn += fn

        total_precision = round(100 * total_tp / (total_tp + total_fp), 2) if total_tp + total_fp != 0 else 0
        total_recall = round(100 * total_tp / (total_tp + total_fn), 2) if total_tp + total_fn != 0 else 0
        total_f1_score = round(2 * total_precision * total_recall / (total_precision + total_recall), 2) if total_precision + total_recall != 0 else 0
        total_stats.append(["total", total_tp, total_fp, total_fn, total_precision, total_recall, total_f1_score])
        cols = ["Index", "True Positive - Good", "False Positive - False", "False Negative - Miss", "Precision", "Recall", "F1 Score"]

        res_df = pd.DataFrame(total_stats, columns=cols).set_index("Index").sort_values(by='True Positive - Good', ascending=False)
        print("result")
        print(res_df)
        self.visualizer.plot_bar_chart(res_df, ["Precision", "Recall", "F1 Score"], "col", dim_col)

        return res_df


    def column_stats(self, cols: list) -> pd.DataFrame:
        return self.df.loc[:, cols].describe()

    
    def cluster_stats(self, ):
        """
        Description: Statistics of training, testing(good/false/miss) for each cluster
        """

        if 'cluster_id' not in self.df.columns:
            self.logger.warning('cluster_id not in dataframe')
            return
        cluster_id_set = list(set(self.df['cluster_id']))
        data_type_set = list(set(self.df['flag']))
        df_cluster_stats = pd.DataFrame(columns = data_type_set)
        for cluster_id in cluster_id_set:
            df_cluster = self.df[self.df['cluster_id'] == cluster_id]
            stats = df_cluster.loc[:, 'flag'].value_counts()
            stats.name = cluster_id
            df_cluster_stats = df_cluster_stats.append(stats, sort=False)
        df_cluster_stats.fillna(0, inplace=True)
        self.visualizer.plot_cluster_stats(df_cluster_stats)


    # def tsne(self, ):
    #     """
    #     Description: T-distributed stochastic neighbor embedding (t-SNE) is a 
    #                  statistical method for visualizing high-dimensional data 
    #                  by giving each datapoint a location in a two or three-
    #                  dimensional map.
    #     """

    #     num_select = VisualizationConfig.MAX_TSNE_SAMPLES \
    #                 if len(self.df) > VisualizationConfig.MAX_TSNE_SAMPLES else len(self.df) 
    #     select_idx = random.sample(list(range(len(self.df))), num_select)
    #     select_emb = self.emb[select_idx]
    #     select_label = self.df['flag'][select_idx].tolist()
    #     self.visualizer.plot_tsne(select_emb, select_label)


    def tsne(self, ):
        """
        Description: T-distributed stochastic neighbor embedding (t-SNE) is a 
                     statistical method for visualizing high-dimensional data 
                     by giving each datapoint a location in a two or three-
                     dimensional map.
        """

        num_select = VisualizationConfig.MAX_TSNE_SAMPLES \
                    if len(self.df) > VisualizationConfig.MAX_TSNE_SAMPLES else len(self.df) 
        select_idx = random.sample(list(range(len(self.df))), num_select)
        
        p0_list = (self.df.priority == 'P0').tolist()
        p0_idx = np.where(np.array(p0_list) == True)[0]
        p0_df_idx = self.df[self.df.priority == 'P0'].index
        select_idx = p0_idx
        
        select_emb = self.emb[select_idx]
        select_label = self.df['flag'][select_idx].tolist()
        
        coor_tsne = self.visualizer.plot_tsne(select_emb, select_label)
        coor_tsne = np.array(coor_tsne)[:,0:2]
        kmeans = KMeans(n_clusters=100, random_state=0, init='k-means++', verbose=0).fit(coor_tsne)
        labels = kmeans.labels_
        self.df['tsne_cluster_id'] = None
        self.df['tsne_coor_x'] = None
        self.df['tsne_coor_y'] = None
        for i, df_idx in enumerate(p0_df_idx):
            self.df.at[df_idx, 'tsne_cluster_id'] = labels[i]
            self.df.at[df_idx, 'tsne_coor_x'],self.df.at[df_idx, 'tsne_coor_y'] = coor_tsne[i].tolist()
        
    
    def t_sne(self, features: np.array, col: str, label_class: str) -> pd.DataFrame:
        print(self.df)
        print(self.df.emb_id)
        print(self.df.cluster_id)
        self.logger.info("TSNE Calculation Started")
        start = time.time()

        tsne = TSNE(n_components=2, init='pca', method='barnes_hut', perplexity=50) 
        X_tsne = tsne.fit_transform(features) 
        X_tsne_data = np.vstack((X_tsne.T, self.dm.get_col_values(col))).T 

        end = time.time()
        self.logger.info("TSNE Calculation Finished")
        self.logger.info("Time Consumed: %.2fs" % (end - start))

        df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'label_class']) 
        df_tsne[['Dim1','Dim2']] = df_tsne[['Dim1','Dim2']].apply(pd.to_numeric)
        df_tsne['cluster_id'] = self.dm.get_col_values(label_class)

        self.logger.info("TSNE Cluster Result Plot Started")
        # pd.to_pickle("/share/analysis/result/syh/dataframes/tsne.pkl")
        # print(df_tsne)
        # df_tsne = pd.read_pickle("/share/analysis/result/syh/dataframes/tsne.pkl")
        self.visualizer.plot_cluster_result(df_tsne)


    def plot_bbox_size(self, ):
        """
        Summary
        -------
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
  
        bbox_size = self.cases_overall_stats("bbox_size")
        self.visualizer.plot_bar_chart(bbox_size, ["Precision", "Recall", "F1 Score"], "col", "bbox_size")


    def my_entropy(self, focus: str = "cluster_id", dim: int = 2, qatrain: bool = False) -> pd.DataFrame:
        """
        Summary
        -------
            0 is the best, 1 is the worst
        Parameters
        ----------
            dim: int
                dim is 2 or 3.
        Returns
        -------
        
        """
        
        def check_none(number):
            return 0 if number == None else number

        uniques = list(self.dm.df[focus].unique())
        stats = self.dm.df.groupby([focus, "flag"]).size()
        stats_dicts = [{unique: {"good": 0, "miss": 0, "false": 0, "train": 0}} for _, unique in enumerate(uniques)]
        for ind, stat in enumerate(stats):
            for dic in stats_dicts:
                dic_key = list(dic.keys())[0]
                stat_index = stats.index[ind]
                if dic_key == stat_index[0]:
                    dic[dic_key][stat_index[1]] = stat
        
        stats_dicts = sorted(stats_dicts, key = lambda _: list(_.keys())[0])
        
        res_lst, ratio_total = [], []
        acc_total, total_good, total_false, total_miss, total_train, total_qa = 0, 0, 0, 0, 0, 0
        

        if not qatrain:
            for each in stats_dicts:
                cluster_id = list(each.keys())[0]
                good = check_none(list(each.values())[0]["good"])
                false = check_none(list(each.values())[0]["false"])
                miss = check_none(list(each.values())[0]["miss"])
                train = check_none(list(each.values())[0]["train"])
                
                if dim == 2:
                    ratio = round(entropy([good, miss + false]), 4)
                elif dim == 3:
                    ratio = round(entropy([good, false, miss]), 4)
                else:
                    self.logger.warning("Please re-selecte the dim")
                res_lst.append([list(each.keys())[0], ratio, good, false, miss])

                if cluster_id != -2:
                    total = good + miss + false
                    acc_total += total
                    ratio_total.append(total * ratio)
                    total_good += good
                    total_false += false
                    total_miss += miss

            df = pd.DataFrame(ratio_total, columns=["ratio_total"]).fillna(0)
            ratio_total = df["ratio_total"].to_list()
            print(ratio_total)
            print(acc_total)
            overall_ratio = round(sum(ratio_total) / acc_total, 4)
            res_lst.append(["overall", overall_ratio, total_good, total_false, total_miss])
            
            df = pd.DataFrame(res_lst, columns=[focus, "my_ratio", "good", "false", "miss"])
            df = df.set_index(focus)
            
            print(df)
            self.visualizer.plot_bar_chart(df, ["my_ratio"], "col", "good_bad_entropy_%s" % focus)
            self.visualizer.plot_stacked_bar(df, "good_bad_stacked_%s" % focus, False)

        else:
            for each in stats_dicts:
                cluster_id = list(each.keys())[0]
                good = check_none(list(each.values())[0]["good"])
                false = check_none(list(each.values())[0]["false"])
                miss = check_none(list(each.values())[0]["miss"])
                train = check_none(list(each.values())[0]["train"])
                
                ratio = round(entropy([train, good + false + miss]), 4)

                res_lst.append([list(each.keys())[0], ratio, train, good + false + miss])

                if cluster_id != -2:
                    total = good + miss + false + train
                    total_train += train
                    total_qa += good + miss + false
                    acc_total += total
                    ratio_total.append(total * ratio)
                    
            
            overall_ratio = round(sum(ratio_total) / acc_total, 4)
            res_lst.append(["overall", overall_ratio, total_train, total_qa])
            
            df = pd.DataFrame(res_lst, columns=[focus, "my_ratio", "train", "qa"])
            df = df.set_index(focus)
            
            print(df)
            self.visualizer.plot_bar_chart(df, ["my_ratio"], "col", "train_qa_entropy_%s" % focus)
            self.visualizer.plot_stacked_bar(df, "train_qa_stacked_%s" % focus, True)
        return df
    
    def run(self, ):
        BaseStats.run(self)
        # self._define_priority()
        # # self.cluster_stats()
        # # # qa_df = dm.load_from_pickle(StatsConfig.EVAL_DATAFRAME_PATH)
        # # # print(qa_df)
        # # self.get_description("cluster_id")
        # # self.get_description("class_name")
        # # self.get_description("size")
        # # good_bad_entropy = self.my_entropy("cluster_id")
        # # qa_train_entropy = self.my_entropy("cluster_id", qatrain=True)
        # # good_bad_entropy = self.my_entropy("class_name")
        # # qa_train_entropy = self.my_entropy("class_name", qatrain=True)
        
        # self.overall_metric()           # precision, recall, and f1-score of P0/P1/ALL
        # self.case_stats()               # number of bad case, miss detection, false detection
        # self.error_analysis()           # false_positive_analysis and false_negative_analysis, add error_type
        # self.define_attribute()
        # self.plot_error_analysis()
        # self.sensitivity_analysis()
        # # self.emb_2_img(sampling=8000, pca_ratio=0.95)
        # self.tsne()

        #TODO commented for new qa format goodcase analysis
        self.compare_multi_version()
        write_json(self.indicator_path, self.indicator_panel)
        temp_dm = dm.DataManager(self.df)
        temp_dm.save_to_pickle(self.df_path)

    def define_attribute(self, ):
        for row in tqdm(self.df.itertuples(), total=len(self.df)):
            index, flag, class_name, matched_id = row.Index, row.flag, row.class_name,row.matched_index
            if matched_id is None:
                continue
            if flag != 'false':
                continue
            try:
                self.df.loc[index, 'occlusion'] = self.df.loc[matched_id, 'occlusion']
                self.df.loc[index, 'crowding'] = self.df.loc[matched_id, 'crowding']
                self.df.loc[index, 'truncation'] = self.df.loc[matched_id, 'truncation']
            except:
                self.logger.error('define 2d attribute error: %s', matched_id)
        temp_dm = dm.DataManager(self.df)
        temp_dm.save_to_pickle(self.df_path)
        
        

    def save_instances(self, cfg: dict):
        index_list = self.dm.getter().index
        bboxs = self.dm.get_cols(["bbox_x", "bbox_y", "bbox_w", "bbox_h"]).values.tolist()
        cluster_id = self.dm.get_col_values("cluster_id")       
        frames = [_.split("@")[0] for _ in index_list]
        instance_id = [_.split("@")[1] for _ in index_list]
        assert len(frames) == len(bboxs)
        
        self.logger.info("Instances Saving Started")
        start = time.time()
        def worker(_):
            with Image.open(_[0]).convert('RGB') as img:
                cropped = img.crop((_[1][0], _[1][1], _[1][0] + abs(_[1][2]), _[1][1] + abs(_[1][3])))
                folder_path = "%s/instances/%d" % (cfg.ROOT, _[2])
                # os.system("rm -rf %s" % folder_path)
                os.makedirs(folder_path, exist_ok=True)
                save_path = "%s/%s_%s.png" % (folder_path, str(_[0].split("/")[-1]).split(".")[0], _[3])
                cropped.save(save_path)
        

        combine_lst = list(map(lambda a, b, c, d: [a, b, c, d], frames, bboxs, cluster_id, instance_id))
        with ThreadPool(processes = 80) as pool:
            self.images = list(tqdm(pool.imap(worker, combine_lst), total=len(combine_lst)))
            pool.terminate()

        end = time.time()
        self.logger.info("Instances Saving Finished")
        self.logger.info("Time Consumed: %.2fs" % (end - start))
    
    def case_stats(self, ):
        # bad case statistic
        case_stats_dic = {'priority':[], 'good':[], 'bad':[], 'miss':[],'miss_proportion':[], \
                      'false':[], 'false_proportion':[]}
        priority_list = ['all', 'P0', 'P1']
        # df_case_stats = pd.DataFrame(columns=stats_name)
        for priority in priority_list:
            if priority == 'all':
                flag_cnt = self.df['flag'].value_counts()
            else:
                flag_cnt = self.df[self.df['priority'] == priority]['flag'].value_counts()
            
            if ('good' not in flag_cnt) or ('false' not in flag_cnt) or ('miss' not in flag_cnt):
                self.logger.error('dataframe[flag] not contain good/false/miss')
                continue
            case_stats_dic['priority'].append(priority)
            case_stats_dic['good'].append(flag_cnt['good'])
            case_stats_dic['bad'].append(flag_cnt['miss']+flag_cnt['false'])
            case_stats_dic['miss'].append(flag_cnt['miss'])
            case_stats_dic['miss_proportion'].append(flag_cnt['miss']/case_stats_dic['bad'][-1])
            case_stats_dic['false'].append(flag_cnt['false'])
            case_stats_dic['false_proportion'].append(flag_cnt['false']/case_stats_dic['bad'][-1])
        df_case_stats = pd.DataFrame(case_stats_dic)
        self.logger.info('\t'+ df_case_stats.to_string().replace('\n', '\n\t')) 

    def overall_metric(self, ):
        """
        Description: computing precision, recall, f1-score for overall/P0/P1
        """
        priority_flag = self.df.groupby(['priority', "flag"]).size().unstack(fill_value=0).to_dict('index')
        # priority_flag = group.to_dict('index')
        if 'P2' in priority_flag:
            priority_flag.pop('P2')
        priority_flag = pd.DataFrame(priority_flag)
        priority_flag['All'] = priority_flag.sum(axis=1)
        priority_flag = priority_flag.T
        flag_list = ['good', 'false', 'miss']
        for flag in flag_list:
            if flag not in priority_flag:
                self.logger.warning("%s not in dataframe, ignore P0 metric."%(flag))
                return
        
        priority_flag['Precision'] = priority_flag['good'] / (priority_flag['good'] +  priority_flag['false'])
        priority_flag['Recall'] = priority_flag['good'] / (priority_flag['good'] +  priority_flag['miss'])
        priority_flag['F1_Score'] = 2 / (1/priority_flag['Precision'] +  1/priority_flag['Recall'])
        
        p0_metrics = priority_flag.loc['P0', ['Precision', 'Recall', 'F1_Score']].to_dict()
        for key, values in p0_metrics.items():
            self.indicator_panel['P0_' + key] = values
        self.logger.info('\t'+ priority_flag.to_string().replace('\n', '\n\t')) 
        return priority_flag
    
    def overall_evaluation(self, ):
        pass

    def sensitivity_analysis(self, ):
        features = ['priority', 'size', 'class_name', 'occlusion', 'truncation', 'crowding', 'direction']
        stats_fetures = {}
        for feature in features:
            train_cnt = self.df[self.df['flag'] == 'train'][feature].value_counts()
            test_cnt = self.df[(self.df['flag'] == 'good') | (self.df['flag'] == 'miss')][feature].value_counts()
            bad_case_cnt = self.df[self.df['flag'] == 'miss'][feature].value_counts()
            stats_fetures[feature] = dict(train=train_cnt.to_dict(), 
                                        test=test_cnt.to_dict(),
                                        bad_case=bad_case_cnt.to_dict())
            self.visualizer.plot_bar(stats_fetures[feature], feature)
        return stats_fetures

    def _define_priority(self, ):
        """
        Description: Redefine priority level P0/P1/P2
        Param: 
        Returns: 
        """
        self.logger.info('Redefine P0/P1/P2')
        vehicle_list = ['car', 'bus', 'truck']
        vru_list = ['pedestrian', 'rider', 'bicycle', 'tricycle']
        static_list = ['trafficCone', 'water-filledBarrier', 'other', 'accident', 'construction']
        
        for row in tqdm(self.df.itertuples(), total=len(self.df)):
            class_name, flag, occlusion, crowding = row.class_name, row.flag, row.occlusion, row.crowding
            pos_x, pos_y = row.x, row.y
            if row.flag != 'train':
                continue
            occlusion = 0 if math.isnan(occlusion) else occlusion
            crowding = 0 if math.isnan(crowding) else crowding
            priority = 'P0'
            
            if abs(pos_x) > 20 or abs(pos_y) > 20:
                priority = 'P1'  
            if int(occlusion) == 2 or int(crowding) == 2:
                priority = 'P1'
            if pd.isna(pos_x):
                priority = 'P1'
            self.df.at[row.Index, 'priority'] = priority 
         
    def error_analysis(self, ):
        """
        Description: false_positive_analysis and false_negative_analysis
        """
        error_type = []
        matched_index_list = []
        gt_dt_info = dict()
        for row in tqdm(self.df.itertuples(), total=len(self.df)):
            index, flag, class_name,iou,x,y,w,h = row.Index, row.flag, row.class_name, row.iou,\
                                            row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h
                                  
            img_path, id = index.split('@')[0], index.split('@')[1]
            if img_path not in gt_dt_info:
                gt_dt_info[img_path] = dict(gt=[], dt=[])
            if flag == 'gt_false':
                flag = 'miss'
            info = [class_name, [x,y,w,h], index, flag]
            if flag == 'good':
                gt_dt_info[img_path]['gt'].append(info)
                # continue
            elif flag == 'miss':
                gt_dt_info[img_path]['gt'].append(info)
            elif flag == 'false':
                gt_dt_info[img_path]['dt'].append(info)

        for row in tqdm(self.df.itertuples(), total=len(self.df)):
            index, flag, class_name = row.Index, row.flag, row.class_name
            bbox = [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h]
            query = [class_name, bbox]
            img_path = index.split('@')[0]
            if flag == 'good':
                error_type.append('good')
                matched_index_list.append(None)
            elif flag == 'miss':
                error_cls, matched_index = self._false_negative_analysis(query, gt_dt_info[img_path]['dt'])
                error_type.append(error_cls)
                matched_index_list.append(matched_index)
            elif flag == 'false':
                error_cls, matched_index = self._false_positive_analysis(query, gt_dt_info[img_path]['gt'])
                error_type.append(error_cls)
                matched_index_list.append(matched_index)
            else:
                error_type.append(None)
                matched_index_list.append(None)
        self.df['error_type'] = error_type
        self.df['matched_index'] = matched_index_list
        temp_dm = dm.DataManager(self.df)
        temp_dm.save_to_pickle(self.df_path)
        
    def _false_positive_analysis(self, query:list, gt_list_info:list)->str:
        """
        Description: From the perspective of prediction results, collect statistics 
                     on accurate detections, class false positives, background false 
                     positives, and position deviations. 
        Param: query:[class_name, bbox], gt_list_info:[[class_name, bbox], ...]
        Returns: error type, matched index
        """
        max_iou = 0
        max_iou_index = -1
        dt_cls, dt_bbox = query
        if len(gt_list_info) == 0:
            return '2d_false_detection', None
        for idx, [gt_cls, gt_bbox, _, _] in enumerate(gt_list_info):
            iou = compute_iou(dt_bbox, gt_bbox)
            if max_iou < iou:
                max_iou = iou
                max_iou_index = idx
        
        if max_iou_index is not None:
            matched_index = gt_list_info[max_iou_index][2]
        
        if max_iou > 0.5:
            if dt_cls == gt_list_info[max_iou_index][0]:
                if gt_list_info[max_iou_index][3] =='good':
                    return '2d_multi_detection', matched_index
                elif gt_list_info[max_iou_index][3] =='miss':
                    return '3d_position_error', matched_index
            else:
                return '2d_class_error', matched_index
        elif max_iou > 0.1 and dt_cls == gt_list_info[max_iou_index][0]:
            return '2d_location_error', matched_index
        elif max_iou > 0.1 and dt_cls != gt_list_info[max_iou_index][0]:
            return '2d_class_error', matched_index
        elif max_iou < 0.1:
            return '2d_false_detection', None
        return None, None
    
    def _false_negative_analysis(self, query:list, dt_list_info:list)->str:
        """
        Description: From the perspective of actual labels, collect statistics 
                     on accurate detections, class false positives, background 
                     false positives, and position deviations.
        Param: query:[class_name, bbox], dt_list_info:[[class_name, bbox, index], ...]
        Returns: error type, matched index
        """
        max_iou = 0
        max_iou_index = None
        gt_cls, gt_bbox = query
        for idx, [dt_cls, dt_bbox, _, _] in enumerate(dt_list_info):
            iou = compute_iou(gt_bbox, dt_bbox)
            if max_iou < iou:
                max_iou = iou
                max_iou_index = idx
        
        if max_iou_index is not None:
            matched_index = dt_list_info[max_iou_index][2]
        
        if max_iou > 0.5:
            if gt_cls != dt_list_info[max_iou_index][0]:
                return '2d_class_error', matched_index
            return '3d_position_error', matched_index
        elif max_iou > 0.1 and dt_cls == dt_list_info[max_iou_index][0]:
            return '2d_location_error', matched_index
        elif max_iou > 0.1 and dt_cls != dt_list_info[max_iou_index][0]:
            return '2d_location_class_error', matched_index
        elif max_iou < 0.1:
            return '2d_miss_detection', None
        return None, None
    
    def plot_error_analysis(self, ):
        if 'error_type' not in self.df:
            self.error_analysis()
        
        # df_fp = self.df[self.df['flag'] == 'false']
        df_fp = self.df[(self.df['flag'] == 'false') & (self.df['priority'] == 'P0')]
        fp_error_counts = df_fp['error_type'].value_counts()
        self.visualizer.plot_pie(data_dict = fp_error_counts.to_dict(), title='false_detection_analysis')
    
        #df_fn = self.df[self.df['flag'] == 'miss']
        df_fn = self.df[(self.df['flag'] == 'miss') & (self.df['priority'] == 'P0')]
        fn_error_counts = df_fn['error_type'].value_counts()
        self.visualizer.plot_pie(data_dict = fn_error_counts.to_dict(), title='miss_detection_analysis')
        pie_info = dict(miss_detection = fn_error_counts.to_dict(), false_detection=fp_error_counts.to_dict())
        write_json(self.save_dir/'pie_chart.json', pie_info)

        self.logger.info('Saving Bad Case')
        class_label = {'miss':{'pred':[], 'gt':[]}, 'false':{'pred':[], 'gt':[]}}
        for row in tqdm(self.df.itertuples(), total=len(self.df)):
            index, class_name, id, flag, priority, error_type, matched_id = row.Index, row.class_name,\
                                                  row.id, row.flag, row.priority, row.error_type, row.matched_index
            # if (flag == 'miss' and priority != 'P0') or (flag != 'false' or priority != 'P0'):
            #     continue
            if (flag != 'miss' and flag != 'false') or priority != 'P0':
                continue

            img_path = index.split('@')[0]
            img_name = Path(row.json_path).stem
            bbox_list = []
            class_name_list = []
            dt_data = []
            gt_data = []
            if flag == 'miss':
                gt_bbox = [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h]
                bbox_list.append(gt_bbox)
                class_name_list.append('gt_' + class_name)
                gt_data.append(dict(class_name=class_name, bbox=gt_bbox))
                if matched_id is not None:
                    dt_bbox = [self.df.at[matched_id, 'bbox_x'], self.df.at[matched_id, 'bbox_y'],
                            self.df.at[matched_id, 'bbox_w'], self.df.at[matched_id, 'bbox_h']]
                    dt_class_name = self.df.at[matched_id, 'class_name']
                    bbox_list.append(dt_bbox)
                    class_name_list.append('pred_' + dt_class_name)
                    if error_type == '2d_class_error':
                        class_label['miss']['gt'].append(class_name)
                        class_label['miss']['pred'].append(dt_class_name)
                    dt_data.append(dict(class_name=dt_class_name, bbox=dt_bbox))
                    

            if flag == 'false':
                if matched_id is not None:
                    gt_bbox = [self.df.at[matched_id, 'bbox_x'], self.df.at[matched_id, 'bbox_y'],
                            self.df.at[matched_id, 'bbox_w'], self.df.at[matched_id, 'bbox_h']]
                    gt_class_name = self.df.at[matched_id, 'class_name']
                    bbox_list.append(gt_bbox)
                    class_name_list.append('gt_' + gt_class_name)
                    if error_type == '2d_class_error':
                        if type(gt_class_name) != type('a'):
                            continue
                        class_label['false']['gt'].append(gt_class_name)
                        class_label['false']['pred'].append(class_name)
                    gt_data.append(dict(class_name=gt_class_name, bbox=gt_bbox))

                dt_bbox = [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h]
                bbox_list.append(dt_bbox)
                class_name_list.append('pred_' + class_name)

                dt_data.append(dict(class_name=class_name, bbox=dt_bbox))
            if error_type is None:
                continue
            info = dict(imgUrl=img_path[1:], error_type=error_type, dt_data=dt_data, gt_data=gt_data)
            save_path = self.save_dir / (flag + '_detection_' + error_type)/ (img_name + '_' + str(int(id))+'.json')
            write_json(save_path, info)
            # save_path = os.path.join('./error_img/', flag, \
            #                '_'.join([str(error_type), class_name, img_name, str(int(id))]) + '.jpg')
            # self.visualizer.draw_bbox(img_path, bbox_list, save_path, text_list = class_name_list)
            # t = threading.Thread(target=self.visualizer.draw_bbox, args=(img_path, bbox_list, save_path, class_name_list))
            # t.start()

        for key, values in class_label.items():
            gt_label = values['gt']
            pred_label = values['pred']
            for i in range(len(gt_label)-1, -1, -1):
                if type(gt_label[i]) != type(pred_label[i]):
                    del class_label[key]['gt'][i]
                    del class_label[key]['pred'][i]
 
            self.visualizer.plot_confusion_matrix(gt_label, pred_label, key)
        
        vehicle_list = ['car', 'bus', 'truck']
        vru_list = ['pedestrian', 'rider', 'bicycle', 'tricycle']
        for key, values in class_label.items():
            gt_label = []
            pred_label = []
            for _ in values['gt']:
                gt_label.append('vehicle') if _ in vehicle_list else gt_label.append('vru')
            for _ in values['pred']:
                pred_label.append('vehicle') if _ in vehicle_list else pred_label.append('vru')
            self.visualizer.plot_confusion_matrix(gt_label, pred_label, key + '_vehicle_vru')     
            

    def emb_2_img(self, sampling: int = 0, pca_ratio: float = 1.0):
        temp_dm = dm.DataManager(df = self.dm.df[self.dm.df['flag'] == "miss"])
        index_list = temp_dm.getter().index
        emb_ids = np.array(temp_dm.get_col_values("emb_id"))
        bboxs = temp_dm.get_cols(["bbox_x", "bbox_y", "bbox_w", "bbox_h"]).values.tolist()       
        frames = [_.split("@")[0] for _ in index_list]
        assert len(frames) == len(bboxs)
        
        # self.logger.info("Images Cropping Started")
        # start = time.time()

        folder_path = "%s/emb_2_img" % (StatsConfig.ROOT)

        # def worker(_):
        #     with Image.open(_[0]).convert('RGB') as img:
        #         cropped = img.crop((_[1][0], _[1][1], _[1][0] + abs(_[1][2]), _[1][1] + abs(_[1][3])))
                
        #         os.makedirs(folder_path, exist_ok=True)
        #         save_path = "%s/%s.png" % (folder_path, str(_[2]))
        #         cropped.save(save_path)
        

        # combine_lst = list(map(lambda a, b, c: [a, b, c], frames, bboxs, emb_ids))
        # with ThreadPool(processes = 80) as pool:
        #     cropped = list(tqdm(pool.imap(worker, combine_lst), total=len(combine_lst)))
        #     pool.terminate()

        # end = time.time()
        # self.logger.info("Instances Saving Finished")
        # self.logger.info("Time Consumed: %.2fs" % (end - start))
        # self.logger.info("Total Images: %s" % str(len(cropped)))

        cropped_lst = []
        for path in os.listdir(folder_path):
            full_path = os.path.join(folder_path, path)
            if os.path.isfile(full_path):
                cropped_lst.append(full_path)
        cropped_lst = np.array(sorted(cropped_lst, key = lambda _ : int(_.split("/")[-1].split(".")[0])))

        if sampling == 0:
            embeddings = np.load(StatsConfig.EMB_PATH)[emb_ids]
        else:
            random_ind = [randint(0, len(cropped_lst)) for _ in range(sampling)]

            emb_ids = emb_ids[random_ind]
            embeddings = np.load(StatsConfig.EMB_PATH)[emb_ids]
            cropped_lst = cropped_lst[random_ind]

        if pca_ratio != 1.0:
            embeddings = PCA(n_components=pca_ratio).fit_transform(embeddings)

        self.logger.info("TSNE Calculation Started")
        start = time.time()

        tsne = TSNE(n_components=2, init='pca', method='barnes_hut', perplexity=50) 
        X_tsne = tsne.fit_transform(embeddings) 

        # tsne = TSNE(
        #     perplexity=50,
        #     metric="euclidean",
        #     n_jobs=8,
        #     random_state=42,
        #     verbose=True,
        # )
        # X_tsne = tsne.fit(embeddings)

        X_tsne_data = np.vstack((X_tsne.T, emb_ids)).T 

        df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'emb_id']) 
        df_tsne[['Dim1','Dim2']] = df_tsne[['Dim1','Dim2']].apply(pd.to_numeric)

        end = time.time()
        self.logger.info("TSNE Calculation Finished")
        self.logger.info("Time Consumed: %.2fs" % (end - start))

        # df_path = "%s/tsne_df.pkl" % folder_path

        # with open(df_path, "wb") as pickle_file: 
        #         pickle.dump(df_tsne, pickle_file)

        # with open(df_path, "rb") as pickle_file:
        #      df_tsne = pickle.load(pickle_file)

        start = time.time()
        self.logger.info("EMB_2_IMAGE Plot Started")

        def getImage(path):
            return OffsetImage(Image.open(path).resize((64, 64)))

        x = df_tsne['Dim1']
        y = df_tsne['Dim2']

        fig = plt.figure(figsize=(80, 80))
        ax = fig.add_subplot()
        ax.scatter(x, y) 
        plt.axis('off')

        for x_0, y_0, path in zip(x, y, cropped_lst):
            ab = AnnotationBbox(getImage(path), (x_0, y_0), frameon=False)
            ax.add_artist(ab)

        fig.savefig("%d.png" % sampling)
        end = time.time()
        self.logger.info("EMB_2_IMAGE Plot Finished")
        self.logger.info("Time Consumed: %.2fs" % (end - start))
        
        
    def label_rules_error(self, ):
        def draw_bbox(img_path:str, bbox:list, save_path = None) -> None:
            """
            Description: save cropped images
            Param:  bbox: x,y,w,h
            Returns: 
            """
            img = Image.open('/' + img_path)
            img_draw = ImageDraw.ImageDraw(img)
            # 在边界框的两点（左上角、右下角）画矩形，无填充，边框红色，边框像素为5
            img_draw.rectangle((bbox[0], bbox[1],bbox[0] + abs(bbox[2]), bbox[1] + abs(bbox[3])), fill=None, outline='red', width=5)  
            shutil.os.makedirs(Path(save_path).parent, exist_ok=True)
            if save_path is not None:
                img.save(save_path)

        errors = ["foreground", "background"]
        fb_df = self.df[(self.df.error_type.isin(errors)) & (self.df.priority == "P0")]
        print(fb_df)
        fb_dm = dm.DataManager(df=fb_df)
        fb_dm.save_to_pickle('%s/dataframes/label_rules_dataframe.pkl' % Config.ROOT)
        
        class ThisClusterConfig(Config):
            DATAFRAME_PATH = '%s/dataframes/label_rules_dataframe.pkl' % Config.ROOT
            CLUSTER_METHOD = "optics"    # kmeans, dbscan...
            PCA_VAR_RATIO = 0.95
            
        fb_cluster = Clustering(ThisClusterConfig)
        fb_cluster.clustering(fb_df.emb_id.to_list(), not_all=False)
        
        fb_df = dm.load_from_pickle(ThisClusterConfig.DATAFRAME_PATH)
        
        false_df = fb_df[fb_df.flag == "false"]
        miss_df = fb_df[fb_df.flag == "miss"]
        total_index = fb_df.groupby("cluster_id").size().index.tolist()
        
        def insert_values(int_lst: list, np_arr: np.array):
            for _ in int_lst:
                np_arr = np.insert(np_arr, _, 0)
            return np_arr

        def get_count(df: pd.DataFrame, total_count_index: np.array) -> np.array:
            count_index = df.groupby(["cluster_id"]).size().index.tolist()
            count = df.groupby(["cluster_id"]).size().to_numpy()
            total_intersection = sorted(list(set(total_count_index) - (set(count_index))))
            return insert_values(total_intersection, count)
        
        false_count = get_count(false_df, total_index)
        miss_count = get_count(miss_df, total_index)
        assert len(false_count) == len(miss_count)

        fb_entropy = entropy([false_count / np.sum(false_count), miss_count / np.sum(miss_count)])
        
        self.logger.info("Instances Saving Started")
        start = time.time()
        def worker(_):
            img_path, case_class, flag, cluster_id = getattr(fb_df, "index")[_].split("@")[0], getattr(fb_df, "class_name")[_], getattr(fb_df, "flag")[_], getattr(fb_df, "cluster_id")[_]
            bbox = [getattr(fb_df, "bbox_x")[_], getattr(fb_df, "bbox_y")[_], getattr(fb_df, "bbox_w")[_], getattr(fb_df, "bbox_h")[_]]
            cluster_entropy = fb_entropy[cluster_id]
            save_folder = "%s/syh/rules_%s/%s_%d/" % (Config.ROOT, ThisClusterConfig.CLUSTER_METHOD, str(cluster_entropy), cluster_id)
            os.makedirs(save_folder, exist_ok=True)
            save_path = "%s%s_%s_%d.png" % (save_folder, flag, case_class, _)
            draw_bbox(img_path, bbox, save_path)
            
        combine_lst = [_ for _ in range(len(fb_df))]
        with ThreadPool(processes = 80) as pool:
            list(tqdm(pool.imap(worker, combine_lst), total=len(combine_lst)))
            pool.terminate()
            
        end = time.time()
        self.logger.info("Instances Saving Finished")
        self.logger.info("Time Consumed: %.2fs" % (end - start))
    
    def compare_multi_version(self, show=True):
        self.logger.info("Start Multi Version Analysis")
        if len(self.df_multi_version) < 2:
            self.logger.warning("There are less than 2 multi version files, multi version comparison is not allowed !")
            return
        self.qa_base_df = self.df_multi_version[0]
        df_base = self.df_multi_version[0]
        dic_multi = collections.OrderedDict(Index=[], flag_base=[])

        dic_others_version = [df.to_dict() for df in self.df_multi_version[1:]]
        for row in tqdm(df_base.itertuples(), total=len(df_base)):
            index,flag = getattr(row, 'Index'), getattr(row, 'flag')
            if (flag != 'good') and (flag != 'miss'):
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
        self._define_case_change(df_multi_comp)
        self.qa_base_df['multi_version_info'] = None
        self.df['multi_version_info'] = None
        for row in tqdm(df_multi_comp.itertuples(), total=len(df_multi_comp)):
            index, multi_version =  getattr(row, 'Index'), getattr(row, 'multi_version')
            try:
                self.qa_base_df.at[index, 'multi_version_info'] = multi_version
                self.df.at[index, 'multi_version_info'] = multi_version
            except:
                print(index)
        temp_dm = dm.DataManager(self.qa_base_df)
        print(df_multi_comp)
  
        priority_multi_info = self.qa_base_df.groupby(['priority', 'multi_version_info']).size().unstack(fill_value=0).to_dict('index')
        p0_miss =  self.qa_base_df[(self.qa_base_df['priority'] == 'P0') & (self.qa_base_df['flag'] == 'miss')].shape[0]
        p0_false =  self.qa_base_df[(self.qa_base_df['priority'] == 'P0') & (self.qa_base_df['flag'] == 'false')].shape[0]
        p0_good = self.qa_base_df[(self.qa_base_df['priority'] == 'P0') & (self.qa_base_df['flag'] == 'good')].shape[0]

        # self.indicator_panel['precision'] = p0_good / (p0_good + p0_false)
        # self.indicator_panel['recall'] = p0_good / (p0_good + p0_miss)
        # self.indicator_panel['f1-score'] = 2/(1/self.indicator_panel['precision'] + 1/ self.indicator_panel['recall'])
        self.indicator_panel['num_p0_miss_case'] = p0_miss
        self.indicator_panel['num_p0_false_case'] = p0_false
        case_discribe_list = ['easy', 'hard', 'fixed', 'retrogression']
        for case in case_discribe_list:
            if case not in priority_multi_info['P0']:
                 priority_multi_info['P0'][case] = 0
        self.indicator_panel['p0_fixed_rate'] = priority_multi_info['P0']['fixed'] / p0_miss
        self.indicator_panel['p0_retrogression_rate'] = priority_multi_info['P0']['retrogression'] / p0_miss
        case_change_in_class = self.qa_base_df[self.qa_base_df['priority'] == 'P0'].groupby(['class_name', 'multi_version_info']).size().unstack(fill_value=0)
        self.logger.info('\t'+ case_change_in_class.to_string().replace('\n', '\n\t')) 
        if 'P0' in priority_multi_info:
            for key, value in priority_multi_info['P0'].items():
                 self.indicator_panel[key] = value
        if 'easy' in self.indicator_panel:
            self.indicator_panel.pop('easy')
        
        if show:
            self.logger.info("Start Saving Bad Cases")
            for row in tqdm(df_multi_comp.itertuples(), total=len(df_multi_comp)):
                index, multi_version =  getattr(row, 'Index'), getattr(row, 'multi_version')
                if multi_version == 'easy':
                    continue
                try:
                    if df_base.at[index, 'priority'] != 'P0':
                        continue
                except:
                    continue
                img_path = index.split('@')[0]
                img_name = Path(img_path).stem
                bbox_list = [[df_base.at[index, 'bbox_x'], df_base.at[index, 'bbox_y'],
                              df_base.at[index, 'bbox_w'],df_base.at[index, 'bbox_h']]]
                flag, class_name, id = df_base.at[index, 'flag'],\
                                       df_base.at[index, 'class_name'],\
                                       df_base.at[index, 'id']

                save_path = os.path.join('/cpfs/output/other/', multi_version+'_case', \
                           '_'.join([flag, class_name, img_name, str(int(id))]) + '.jpg')
                class_name_list = [class_name]
                self.visualizer.draw_bbox(img_path, bbox_list, save_path, text_list = class_name_list)
                t = threading.Thread(target=self.visualizer.draw_bbox, args=(img_path, bbox_list, save_path, class_name_list))
                t.start()
        self.logger.info("Multi Version Analysis Finished")
    
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
        

if __name__ == '__main__':
    # temp_dm = dm.DataManager(df=dm.load_from_pickle(StatsConfig.DATAFRAME_PATH))
    # temp_stat = Stats(df=temp_dm.df)
    # # temp_stat.emb_2_img(sampling=8000, pca_ratio=0.95)
    # temp_stat.plot_all()

    # # path = "/share/analysis/result/syh/dataframes/eval_dataframe_kmeans_0.95.pkl"
    # temp_dm = dm.DataManager(df=dm.load_from_pickle(StatsConfig.DATAFRAME_PATH))
    # print(temp_dm.df)
    # print(temp_dm.df.emb_id)
    # temp_stat = StatsCam3D(df_path = StatsConfig.DATAFRAME_PATH, emb_path=StatsConfig.EMB_PATH)
    # print(StatsConfig)
    temp_stat = StatsCam3D(config=StatsConfig)
    # emb = np.load(StatsConfig.EMB_PATH)
    # temp_stat = StatsCam2D(config=StatsConfig)

    # temp_stat.t_sne(emb, "flag", "cluster_id")
    # temp_stat.save_instances()
    temp_stat.run()
    # temp_stat.compare_multi_version()
