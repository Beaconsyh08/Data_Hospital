#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   error_analysis.py
@Time    :   2022/03/30 14:20:10
@Author  :   chenminghua 
'''
import os
import copy
import math
from matplotlib.pyplot import flag
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from src.classification.clustering import Clustering
from typing import List, Dict, Tuple
import threading
from tqdm import tqdm
from configs.config import Config, StatsConfig, VisualizationConfig, OutputConfig
import src.data_manager.data_manager as dm
from src.stats.stats_common import *
from src.visualization.visualization import Visualizer
from src.utils.common import compute_iou
from src.utils.file_io import write_json
from src.utils.logger import get_logger
from src.utils.struct import Obstacle
from src.utils.logger import get_logger
from src.utils.struct import Obstacle, parse_obs

class ErrorAnalysis(object):
    def __init__(self, df: pd.DataFrame = None, emb = None) -> None:
        self.df = df
        self.emb = emb
        self.logger = get_logger()
        self.save_dir = Path(OutputConfig.OTHERS_OUTPUT_DIR)
        self.visualizer = Visualizer(VisualizationConfig)

    def load_data(self, df: pd.DataFrame = None, df_path: str = None, emb = None):
        """
        load dataframe
        """
        self.df = df
        self.all_gt_data = {}
        self.all_data = {}
        self.emb = emb
        self.df_p0 = self.df[self.df.priority == 'P0']
        for row in self.df.itertuples():
            index, flag = row.Index, row.flag
            imgUrl = index.split('@')[0]
            
            if imgUrl not in self.all_data:
                self.all_data[imgUrl] = []
            self.all_data[imgUrl].append(index)
            if flag not in ['miss', 'good']:
                continue
            if imgUrl not in self.all_gt_data:
                self.all_gt_data[imgUrl] = []
            self.all_gt_data[imgUrl].append(index)


        # pdb.set_trace()
    def _define_error_type(self, ):
        self.df['error_type'] = None
        # self.pre_process()
        for row in tqdm(self.df.itertuples(), total=len(self.df), desc='Define error type'):
            index, flag = row.Index, row.flag

            if row.priority != 'P0':
                continue
            if flag == 'good':
                self.df.at[index, 'error_type'] = None
                continue
            elif flag == 'miss':
                self.df.at[index, 'error_type'] = self._miss_analysis(row, )
            elif flag == 'false':
                self.df.at[index, 'error_type'] = self._false_analysis(row, )
        
        temp_dm = dm.DataManager(self.df)
        temp_dm.save_to_pickle(Config.DATAFRAME_PATH)

    def process(self, ):
        """
        Description: false_positive_analysis and false_negative_analysis
        """
        self.logger.info('start error analysis')
        self._define_error_type()
        
        self.logger.info('start sensitivity analysis')
        # self.bev_truncation_acc()
        self.sensitivity_analysis()
        # self._define_long_tail()
        self.plot_error_pie()
        # self.tsne()
        self.save_error_case()
        # self.class_error_analysis()
        self.output_topk_error()
        temp_dm = dm.DataManager(self.df)
        temp_dm.save_to_pickle(Config.DATAFRAME_PATH)
    
    def show_result(self, ):
        pass
   
    # def _define_long_tail(self, ):
    #     self.df_p0 = self.df[self.df.priority == 'P0']
    #     cluster_id_list = set(self.df_p0.cluster_id)
    #     self.df['long_tail_score'] = None
    #     num_train = (self.df.flag == 'train').sum()
    #     num_test = (self.df.flag != 'train').sum()
    #     for cluster_id in tqdm(cluster_id_list, 'Define long tail'):
    #         df_cluster = self.df_p0[self.df_p0.cluster_id == cluster_id]
    #         cluster_flag_dic = df_cluster.groupby(['flag']).size().to_dict()
    #         num_train, num_good =  cluster_flag_dic.get('train', 0), cluster_flag_dic.get('good', 0)
    #         num_miss, num_false = cluster_flag_dic.get('miss', 0),cluster_flag_dic.get('false', 0)
    #         long_tail_score = (num_miss + num_false) / (num_train  + 0.1)
    #         self.df.loc[self.df[(self.df.cluster_id == 0) & (self.df.priority == 'P0')].index.tolist(), 'long_tail_score'] = long_tail_score
    #         for row in df_cluster.itertuples():
    #             obs_list = [parse_obs(row)]
    #             if not pd.isna(row.peer_id):
    #                 obs_list.append(parse_obs(self.df.loc[row.peer_id]))
    #             save_path = Path(os.path.join('long_tail_score', str("%.2f" % long_tail_score) +  '_' + str(len(df_cluster)) + '_' +  str(row.cluster_id),\
    #                                           '_'.join([row.flag, str(row.error_type), str(Path(row.json_path).stem)+'_'+str(row.id)+'.jpg'])))
    #             # save_path = Path(os.path.join('long_tail_score', '_'.join([row.flag, str(row.error_type),row.class_name, str(Path(row.json_path).stem)+'_'+str(row.id)+'.jpg'])))
    #             if pd.isna(row.crowding) and row.flag ==  'miss': 
    #                 bev_save_path = save_path.parent / (save_path.stem + '_bev.jpg')
    #                 t0 = threading.Thread(target=self.visualizer.draw_bbox_0, args=(obs_list, str(save_path)))
    #                 t0.start()
    #                 t1 = threading.Thread(target=self.visualizer.plot_bird_view, args=(obs_list, str(bev_save_path)))
    #                 t1.start()

    def pre_process(self, ):
        flag_dic = {'miss':'1', 'false': '2'}
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        for row in tqdm(self.df.itertuples(), total=len(self.df), desc='pre_processing'):
            if row.case_flag != '21' or pd.isna(row.peer_id) or row.priority != 'P0':
                continue

            curr_bbox = [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h]
            peer_row = self.df.loc[row.peer_id]
            peer_bbox = [peer_row.bbox_x, peer_row.bbox_y, peer_row.bbox_w, peer_row.bbox_h]
            if compute_iou(curr_bbox, peer_bbox) < 0.1 and row.flag in flag_dic:
                self.df.at[row.Index, 'case_flag'] = flag_dic[row.flag]

    def check_overlap(self, row):
        imgUrl =  row.Index.split('@')[0]
        bbox_0 = [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h]
        if imgUrl not in self.all_gt_data:
            # self.logger.warning("%s not in all_gt_data" %(imgUrl))
            return False
        for idx in self.all_gt_data[imgUrl]:
            if idx == row.Index:
                continue
            info = self.df.loc[idx] if type(self.df.loc[idx]) == pd.Series else self.df.loc[idx].iloc[0]
            bbox_1 = [info.bbox_x, info.bbox_y, info.bbox_w, info.bbox_h]
            # print(bbox_0, bbox_1, compute_iou(bbox_0, bbox_1))
            if compute_iou(bbox_0, bbox_1) > 0 and row.self_dis > info.self_dis:
                return True
            if np.linalg.norm(np.array([row.x, row.y]) - np.array([info.x, info.y])) < 1:
                return True
        
        return False

    def _check_2d_miss(self, row) -> str:

        bbox = [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h]
        location = [row.x, row.y]
        obstacle = parse_obs(row)
        if row.broad_category == 'vehicle':
            truncation_info = check_truncation(obstacle)
            if truncation_info:
                return '2d_miss_vehicle_' + truncation_info
            if row.class_name == 'truck':
                return '2d_miss_vehicle_truck'
            return '2d_miss_vehicle_normal'
        elif row.broad_category == 'vru':
            return '2d_miss_vru_group' if self.check_overlap(row) else '2d_miss_vru_normal'

    def _check_3d_error(self, row):
        bbox = [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h]
        location = [row.x, row.y]
        obstacle = parse_obs(row)
        if row.broad_category == 'vehicle':
            truncation_info = check_truncation(obstacle)
            if truncation_info:
                return '3d_position_vehicle_' + truncation_info
            if (row.yaw>3.14/4 and row.yaw<3.14*3/4) or (-row.yaw > 3.14/4 and -row.yaw<3.14*3/4):
                return '3d_position_vehicle_rotation' 

            return '3d_position_vehicle_normal'
        elif row.broad_category == 'vru':
            return '3d_position_vru'

 
    def _miss_analysis(self, row) -> str:
        """
        miss detection analysis
        """
        error_type = None
        flag_2d, flag_3d= row.case_flag, row.bev_case_flag
        if flag_2d == '0':
            return 'good' if flag_3d == '0' else self._check_3d_error(row)
        elif flag_2d == '1':
            return self._check_2d_miss(row)
        elif flag_2d == '21':
            return '2d_location_error'
        elif flag_2d == '22':
            return '2d_class_error'
        else:
            return 'others'
    
    def _check_2d_false(self,row):
        obstacle = parse_obs(row)
        if self.check_overlap(row):
            return '2d_false_group'
        truncation_info = check_truncation(obstacle)
        if truncation_info:
            return '2d_false_' + truncation_info
        return '2d_false_label_error'
    
    def _false_analysis(self, row) -> str:
        """
        false detection analysis
        """
        error_type = None
        flag_2d, flag_3d= row.case_flag, row.bev_case_flag
        obstacle = parse_obs(row)
        if flag_2d == '0':
            return 'good' if flag_3d == '0' else self._check_3d_error(row)
        elif flag_2d == '2':
            return self._check_2d_false(row)
        elif flag_2d == '21':
            return '2d_location_error'
        elif flag_2d == '22':
            return '2d_class_error'
        else:
            return 'others'
    
    def class_error_analysis(self, ):
        gt_labels = []
        dt_labels = []
        df_p0 = self.df[self.df.priority == 'P0']
        for row in tqdm(df_p0.itertuples(), total=len(df_p0)):
            class_name, peer_id, dtgt = row.class_name, row.peer_id, row.dtgt
            if dtgt == 'dt' or pd.isna(peer_id):
                continue
            
            peer_row = self.df.loc[peer_id] if type(self.df.loc[peer_id]) == pd.Series \
                            else self.df.loc[peer_id].iloc[0]
            dt_cls = peer_row.class_name
            if class_name == dt_cls:
                continue
            gt_labels.append(class_name)
            dt_labels.append(dt_cls)
        self.visualizer.plot_confusion_matrix(gt_labels, dt_labels, 'miss_detection')

    # def parse_obs(self, row):
    #     # if type(row) != pd.Series:
    #     #     self.logger.warning("type(row) != pd.Series, parse obs error")
    #     #     return
    #     obs = Obstacle()
    #     try:
    #         obs.img_path = row.name.split('@')[0]
    #     except Exception:
    #         obs.img_path = row.Index.split('@')[0]
    #     obs.id, obs.dtgt = row.id, row.dtgt
    #     obs.flag =  row.flag
    #     obs.bbox = [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h]
    #     obs.position = [row.x, row.y, row.z]
    #     obs.x, obs.y, obs.z = obs.position
    #     obs.height, obs.length, obs.width = row.height, row.length, row.width
    #     obs.class_name, obs.yaw = row.class_name, row.yaw
    #     obs.truncation, obs.crowding, obs.occlusion = row.truncation, row.crowding, row.occlusion
    #     return obs
    
    def save_error_case(self, show=True):
        """
        Save all bad cases(false detection, miss detection) in /cpfs/output/other
        The JSON file contains: imageUrl, gt_info, dt_info
        """
        self.logger.info(f"Saving error case in: {self.save_dir}")
        for row in tqdm(self.df.itertuples(), total=len(self.df), desc="Saving error case"):
            index, id, class_name, flag, priority, error_type = row.Index, row.id, row.class_name, \
                                                       row.flag, row.priority, row.error_type
            if flag not in ['miss', 'false'] or priority != 'P0':
                continue
            img_path = index.split('@')[0]
            img_name = Path(row.json_path).stem
            bbox_list, class_name_list = [], []
            gt_data, dt_data = [], []
            curr_bbox, curr_class_name = [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h], str(row.class_name)
            if curr_class_name == 'static':
                continue

            obs_list = [parse_obs(row)]
            if not pd.isna(row.peer_id):
                if row.peer_id not in self.df.index:
                    continue
                peer_row = self.df.loc[row.peer_id] if type(self.df.loc[row.peer_id]) == pd.Series else self.df.loc[row.peer_id].iloc[0]
                peer_bbox = [peer_row.bbox_x, peer_row.bbox_y, peer_row.bbox_w, peer_row.bbox_h]
                peer_class_name = peer_row.class_name
                obs_list.append(parse_obs(peer_row))


            if flag == 'miss':
                gt_data.append(dict(class_name = curr_class_name, bbox=curr_bbox))
                bbox_list.append(curr_bbox)
                class_name_list.append('gt_' + curr_class_name)
                if not pd.isna(row.peer_id):
                    dt_data.append(dict(class_name = peer_class_name, bbox=peer_bbox)) 
                    bbox_list.append(peer_bbox)
                    class_name_list.append('dt_' + peer_class_name)

            elif flag == 'false':
                dt_data.append(dict(class_name = curr_class_name, bbox=curr_bbox))
                bbox_list.append(curr_bbox)
                class_name_list.append('dt_' + curr_class_name)
                if not pd.isna(row.peer_id):
                    gt_data.append(dict(class_name = peer_class_name, bbox=peer_bbox))
                    bbox_list.append(peer_bbox)
                    class_name_list.append('gt_' + peer_class_name)
            if error_type is None:
                continue
            info = dict(imgUrl=img_path[1:], error_type=error_type, dt_data=dt_data, gt_data=gt_data)
            save_path = self.save_dir / (flag + '_detection_' + error_type)/ (img_name + '_' + str(int(id))+'.json')
            write_json(save_path, info)
            frame_obs_list = []
            if img_path in self.all_data:
                for idx in self.all_data[img_path]:
                    new_row = self.df.loc[idx] if type(self.df.loc[idx]) == pd.Series else self.df.loc[idx].iloc[0]
                    obs = parse_obs(new_row)
                    frame_obs_list.append(obs)
                    if obs.flag == 'good':
                        obs_cp = copy.deepcopy(obs)
                        obs_cp.dtgt = 'dt'
                        frame_obs_list.append(obs_cp)

            if show:
                save_path = os.path.join('./error_img/', flag, \
                           '_'.join([str(error_type), class_name, img_name, str(int(id))]) + '.jpg')
                # self.visualizer.draw_bbox(img_path, bbox_list, save_path, text_list = class_name_list)
                # self.visualizer.plot_bird_view(obs_list, Path(save_path).parent/(str(Path(save_path).stem)+'_bev.jpg'))
                bev_save_path = str(Path(save_path).parent/(str(Path(save_path).stem)+'_bev.jpg'))
                t0 = threading.Thread(target=self.visualizer.draw_bbox_0, args=(obs_list, save_path))
                t0.start()
                t1 = threading.Thread(target=self.visualizer.plot_bird_view, args=(obs_list, bev_save_path))
                t1.start()

                frame_save_path = str(Path(save_path).parent/(str(Path(save_path).stem)+'_frame.jpg'))
                frame_bev_save_path = str(Path(save_path).parent/(str(Path(save_path).stem)+'_frame_bev.jpg'))
                t2 = threading.Thread(target=self.visualizer.draw_bbox_0, args=(frame_obs_list, frame_save_path))
                t2.start()
                t3 = threading.Thread(target=self.visualizer.plot_bird_view, args=(frame_obs_list, frame_bev_save_path))
                t3.start()

    def plot_error_pie(self, ):
        # if 'error_type' not in self.df:
        #     self.error_analysis()
        
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
    
    def sensitivity_analysis(self, ):
        """
        Show the distribution of each feature in train/test/miss datasets
        """
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
    
    def bev_truncation_acc(self, ):
        cnt_truncation = 0
        cnt_truncation_acc = 0
        self.all_json_data = self.df.json_path.to_list()
        for row in tqdm(self.df.itertuples(), total=len(self.df), desc="computing bev truncation acc"):
            if row.flag not in ['miss', 'false'] or row.priority != 'P0':
                continue
            if 'truncation' not in row.error_type:
                continue
            # img_list = list(self.all_gt_data.keys())
            json_name = Path(row.json_path).stem.split('_')[-1]
            # find_path = []
            cnt_truncation+=1
            idx_bev_acc = None
            for idx, path in enumerate(self.all_json_data):
                if json_name not in path:
                    continue
                # print(img_name, path, )
                other_orientation_row = self.df.iloc[idx]
                if np.linalg.norm(np.array([row.x, row.y]) - np.array([other_orientation_row.x, other_orientation_row.y])) < 0.1 \
                    and other_orientation_row.flag == 'good':
                    cnt_truncation_acc+=1
                    idx_bev_acc = other_orientation_row.name
                    break

            if idx_bev_acc is None:
                continue
            bev_acc_row = self.df.loc[idx_bev_acc]
            bev_acc_row.dtgt = 'dt'
            obs_list = [parse_obs(bev_acc_row)]
            # try:
            #     obs_list = [parse_obs(bev_acc_row), \
            #                 parse_obs(self.df.loc[bev_acc_row.peer_id])]
            # except:
            #     continue
            show = True
            if show:
                save_path = os.path.join('./error_img/', row.flag, \
                           '_'.join([str(row.error_type), row.class_name, Path(row.json_path).stem, str(int(row.id))]) + '_g.jpg')
                # self.visualizer.draw_bbox(img_path, bbox_list, save_path, text_list = class_name_list)
                # self.visualizer.plot_bird_view(obs_list, Path(save_path).parent/(str(Path(save_path).stem)+'_bev.jpg'))
                bev_save_path = str(Path(save_path).parent/(str(Path(save_path).stem)+'_bev.jpg'))
                t0 = threading.Thread(target=self.visualizer.draw_bbox_0, args=(obs_list, save_path))
                t0.start()
                t1 = threading.Thread(target=self.visualizer.plot_bird_view, args=(obs_list, bev_save_path))
                t1.start()

        self.logger.info("truncation acc: %d/%d=%f"%(cnt_truncation_acc, cnt_truncation, cnt_truncation_acc/cnt_truncation))


                
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
    
    def output_topk_error(self, ):
        # topk_pos_error_list = self._top_k_position_error(top_k=10)
        # topk_yaw_error_list = self._top_k_yaw_error(top_k=10)
        topk_error_dic = dict(pos_error=self._top_k_position_error(top_k=100), 
                              yaw_error=self._top_k_yaw_error(top_k=100), 
                              normal_vehicle=self._top_k_normal_vehicle_error(top_k=100),
                              closed_vehicle_yaw=self._find_closed_vehicle(),
                              normal_2dvru_error=self._top_k_normal_2dvru_error())
        for key, values in tqdm(topk_error_dic.items(), desc="Saving Topk Error"):
            for i, idx in enumerate(values):
                row = self.df.loc[idx] if type(self.df.loc[idx]) == pd.Series else self.df.loc[idx].iloc[0]
                obs = parse_obs(row)
                img_name, id = Path(row.name).stem, row.id
                obs_list = [parse_obs(row)]
                if not pd.isna(row.peer_id) and row.peer_id in self.df.index:
                    peer_row = self.df.loc[row.peer_id] if type(self.df.loc[row.peer_id]) == pd.Series else self.df.loc[row.peer_id].iloc[0]
                    obs_list.append(parse_obs(peer_row))
                save_path = os.path.join('./topk_error/', key, \
                            '_'.join([str(i), str(row.truncation),str(row.error_type), row.class_name, img_name, str(int(id))]) + '.jpg')
                bev_save_path = str(Path(save_path).parent/(str(Path(save_path).stem)+'_bev.jpg'))
                t0 = threading.Thread(target=self.visualizer.draw_bbox_0, args=(obs_list, save_path))
                t0.start()
                t1 = threading.Thread(target=self.visualizer.plot_bird_view, args=(obs_list, bev_save_path))
                t1.start()

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
        kmeans = KMeans(n_clusters=int((len(coor_tsne)/2)**0.5), random_state=0, init='k-means++', verbose=1).fit(coor_tsne)
        labels = kmeans.labels_
        self.df['tsne_cluster_id'] = None
        self.df['tsne_coor_x'] = None
        self.df['tsne_coor_y'] = None
        for i, df_idx in enumerate(tqdm(p0_df_idx, total=len(p0_df_idx))):
            self.df.at[df_idx, 'tsne_cluster_id'] = labels[i]
            # self.df.at[df_idx, 'tsne_coor_x'],self.df.at[df_idx, 'tsne_coor_y'] = coor_tsne[i].tolist()
        
    def _top_k_position_error(self, top_k: int = 100):
        df_p0 = self.df[((self.df['flag'] == 'miss') & (self.df['case_flag']=='0')) & (self.df['priority'] == 'P0')]
        df_p0.sort_values("dis_ratio",inplace=True, ascending=False)
        top_k = min(top_k, len(df_p0))
        return df_p0.index[:top_k].to_list()
    
    def _top_k_yaw_error(self, top_k: int = 100):
        df_p0 = self.df[((self.df['flag'] == 'miss') & (self.df['case_flag']=='0')) \
             & (self.df['broad_category'] == 'vehicle') & (self.df['priority'] == 'P0')]
             
        df_p0.sort_values("yaw_diff",inplace=True, ascending=False)
        top_k = min(top_k, len(df_p0))
        return df_p0.index[:top_k].to_list()
    
    def _top_k_normal_vehicle_error(self, top_k: int = 100):
        df_p0 = self.df[(self.df['flag'] == 'miss') & (self.df['broad_category'] == 'vehicle')\
                       & (self.df['truncation'] < 0.5) & (self.df['priority'] == 'P0')]

        # df_p0.sort_values("yaw_diff",inplace=True, ascending=False)
        top_k = min(top_k, len(df_p0))
        return df_p0.index[:top_k].to_list()
    
    def _top_k_normal_2dvru_error(self, top_k: int = 100):
        df_p0 = self.df[(self.df['flag'] == 'miss') & (self.df['broad_category'] == 'vru')\
                       & (self.df['class_name'] != 'bicycle') & (self.df['priority'] == 'P0')]

        # df_p0.sort_values("yaw_diff",inplace=True, ascending=False)
        top_k = min(top_k, len(df_p0))
        return df_p0.index[:top_k].to_list()
    
    def _find_closed_vehicle(self, ):
        df_p0 =  self.df[(self.df['flag'] != 'false') & (self.df['priority'] == 'P0')\
            & (self.df['broad_category'] == 'vehicle') & (abs(self.df.x) < 10)]
        df_p0 = df_p0[((abs(df_p0.yaw) > 5/180*3.14) & (abs(df_p0.yaw) < 20/180*3.14 )) |\
                 (((abs(df_p0.yaw) - 3.14)  > 5/180*3.14) & ((abs(df_p0.yaw) - 3.14) < 20/180*3.14))]
        return df_p0.index.to_list()