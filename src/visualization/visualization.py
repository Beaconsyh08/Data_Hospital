#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   visulization.py
@Time    :   2021/11/15 22:13:20
@Email   :   songyuhao@haomo.ai
@Author  :   YUHAO SONG 

Copyright (c) HAOMO.AI, Inc. and its affiliates. All Rights Reserved
"""
from turtle import color
import cv2
from os import mkdir
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.shape_base import tile
from openTSNE import TSNE
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix
import seaborn as sns
import shutil
import pickle
import time
from typing import List, Dict, Tuple
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
from src.utils.logger import get_logger
from src.utils.struct import Obstacle
from src.visualization.top_viewer import TopViewer



class Visualizer(object):
    def __init__(self, VisualizationConfig) -> None:
        super().__init__()
        self.save_dir = Path(VisualizationConfig.SAVE_DIR)
        shutil.os.makedirs(self.save_dir, exist_ok=True)
        self.logger = get_logger()

    def plot_cluster_stats(self, df: pd.DataFrame) -> None:
        """
        Description: Statistics of training, testing(good/false/miss) for each cluster
        """
        # df.plot(figsize=(20,10))
        ax = df.plot(figsize=(20, 10))
        ax.set_title('Cluster statistics')
        ax.set_xlabel('cluster id')
        ax.set_ylabel('Frequency')
        ax.grid()
        fig = ax.get_figure()
        save_path = self.save_dir / 'cluster_stats.png'
        fig.savefig(save_path)
        self.logger.info("Cluster Stats Plot Saved: %s" % save_path)

    def plot_tsne(self, X, y) -> None:
        """
        Description: T-distributed stochastic neighbor embedding (t-SNE) is a 
                     statistical method for visualizing high-dimensional data 
                     by giving each datapoint a location in a two or three-
                     dimensional map.
        """
        self.logger.critical("TSNE Calculation Started")
        start = time.time()

        tsne = TSNE(
            perplexity=30,
            metric="euclidean",
            n_jobs=8,
            random_state=42,
            verbose=True,
        )
        X_tsne = tsne.fit(X)

        # # save pickle to fix coordinate
        # pkl_file = open('tsne.pkl', 'wb')
        # pickle.dump(X_tsne, pkl_file)
        
        # # load pickle
        # pkl_file = open('tsne.pkl', 'rb')
        # X_tsne = pickle.load(pkl_file)

        X_tsne_data = np.vstack((X_tsne.T, y)).T
        df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])
        df_tsne[['Dim1', 'Dim2']] = df_tsne[[
            'Dim1', 'Dim2']].apply(pd.to_numeric)

        end = time.time()
        self.logger.info("TSNE Calculation Finished")
        self.logger.info("Time Consumed: %.2fs" % (end - start))

        fig = plt.figure(figsize=(15, 15)) 
        color_dict = dict({'train':'green',
                  'good':'blue',
                  'false': 'orange',
                  'miss': 'red',
                  'easy':'green',
                  'hard':'red',
                  'fixed':'blue',
                  'retrogression': 'darkviolet'
                   })
        # ax.set_title('T-distributed stochastic neighbor embedding')
        # sns.scatterplot(data=df_tsne, hue='class',style='class',palette=palette, alpha = 0.8, x='Dim1', y='Dim2') 
        flag_color = True
        for d in set(y):
            if d not in color_dict:
                flag_color = False
        if flag_color:
            sns.scatterplot(data=df_tsne, hue='class',style='class', palette=color_dict, alpha = 0.6, x='Dim1', y='Dim2') 
        else:
             sns.scatterplot(data=df_tsne, hue='class',style='class', alpha = 0.6, x='Dim1', y='Dim2') 
        save_path = self.save_dir / 'tsne.png'
        plt.savefig(save_path)
        self.logger.debug("TSNE Plot Saved: %s" % save_path)
        return df_tsne

    def plot_cluster_result(self, df_tsne) -> None:
        """
        Summary
        -------
            Visualize the clustering result
        Parameters
        ----------

        Returns
        -------

        """

        fig = plt.figure(figsize=(30, 30))

        label_names = df_tsne.label_class.unique()
        colour = ['black', 'grey', 'coral', 'peru', 'lawngreen', 'yellow', 'gold', 'cyan',
                  'fuchsia', 'indigo', 'teal', 'darkgreen', 'aqua', 'cornsilk', 'royalblue', 'tomato']
        colour = colour[:len(df_tsne.label_class.unique())]
        colours = dict(zip(label_names, colour))
        ccc = df_tsne.label_class.map(colours)
        points_plt = plt.scatter(
            df_tsne.Dim1, df_tsne.Dim2, s=15, c=ccc, edgecolors='black', linewidths=0.3)

        # plot centers
        # plt.scatter(cen_x, cen_y, marker='^', c=colors, s=70

        recs = []
        for i in range(0, len(colour)):
            recs.append(mpatches.Circle((0.5, 0.5), radius=0.25, fc=colour[i]))
        plt.legend(recs, label_names, loc=4)

        for i in df_tsne.cluster_id.unique():
            if i == -1:
                continue
            print(i)
            points = df_tsne[df_tsne.cluster_id == i][['Dim1', 'Dim2']].values
            # get convex hull
            hull = ConvexHull(points)
            # get x and y coordinates
            # repeat last point to close the polygon
            x_hull = np.append(points[hull.vertices, 0],
                               points[hull.vertices, 0][0])
            y_hull = np.append(points[hull.vertices, 1],
                               points[hull.vertices, 1][0])
            # plot shape
            plt.fill(x_hull, y_hull, alpha=0.3, c='black')

        # points_plt.set_urls(imgs_paths)
        # folder_path = VisualizationConfig.ROOT + "/" + "vis"
        # if folder_path:
        #     mkdir(folder_path)
        # fig.savefig(folder_path + "/cluster_vis.png")
        fig.savefig("/root/2d_analysis/cluster_vis.jpg")
        self.logger.info("TSNE Cluster Result Plot Finished")

    def crop_and_display(self, ) -> None:
        pass

    def plot_bar_chart(self, df: pd.DataFrame, target_lst: list, row_or_col: str, save_name: str) -> None:
        # TODO put in vis mod
        """
        Summary
        -------
            Plot the bar chat

        Parameters
        ----------
            target_lst: list
                which cols or rows to be plot
            row_or_col: str
                the dimension to plot the chart
        """
        if row_or_col == "col":
            temf_df = df.loc[:, target_lst]
        elif row_or_col == "row":
            temf_df = df.loc[target_lst, :]

        ax = temf_df.plot.bar(fontsize=12, figsize=(15, 10))
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x()
                                              * 1.005, p.get_height() * 1.005))
        save_path = str(self.save_dir) + "/%s.png" % save_name
        ax.get_figure().savefig(save_path)
        self.logger.debug("PRF Plot Saved: %s" % save_path)

    def plot_stacked_bar(self, df: pd.DataFrame, save_name: str, qatrain: bool = True):
        if qatrain:
            train = df["train"].to_numpy()[:-1]
            qa = df["qa"].to_numpy()[:-1]
            total = train + qa
            qa_rate = qa / total
            train_rate = train / total

            labels = [str(_) for _ in df.index.to_numpy()[:-1]]
            labels, train, qa, total = zip(
                *sorted(zip(labels, train, qa, total), key=lambda _: _[3]))

            fig = plt.figure(figsize=(30, 20))
            ax = fig.add_subplot()

            ax.bar(labels, train, label='train')
            ax.bar(labels, qa, bottom=train, label='qa')

            ax.set_ylabel('Amount')
            ax.set_title('Train and QA')
            ax.legend()

            save_path = str(self.save_dir) + "/%s.png" % save_name
            ax.get_figure().savefig(save_path)
            self.logger.debug("Stacked Bar Plot Saved: %s" % save_path)
            plt.close()

            labels = [str(_) for _ in df.index.to_numpy()[:-1]]
            labels, train_rate, qa_rate = zip(
                *sorted(zip(labels, train_rate, qa_rate), key=lambda _: _[2]))

            fig = plt.figure(figsize=(30, 20))
            ax = fig.add_subplot()

            ax.bar(labels, train_rate, label='train_rate')
            ax.bar(labels, qa_rate, bottom=train_rate, label='qa_rate')

            ax.set_ylabel('Rate')
            ax.set_title('Train and QA')
            ax.legend()

            save_path = str(self.save_dir) + "/%s_rate.png" % save_name
            ax.get_figure().savefig(save_path)
            self.logger.debug("Stacked Bar Plot Saved: %s" % save_path)
            plt.close()

        else:
            good = df["good"].to_numpy()[:-1]
            false = df["false"].to_numpy()[:-1]
            miss = df["miss"].to_numpy()[:-1]
            total = good + false + miss
            np.seterr(divide='ignore', invalid='ignore')
            good_rate = good / total
            false_rate = false / total
            miss_rate = miss / total
            good_rate[np.isnan(good_rate)] = 0
            false_rate[np.isnan(false_rate)] = 0
            miss_rate[np.isnan(miss_rate)] = 0

            labels = [str(_) for _ in df.index.to_numpy()[:-1]]
            labels, good, false, miss, total = zip(
                *sorted(zip(labels, good, false, miss, total), key=lambda _: _[4]))

            fig = plt.figure(figsize=(30, 20))
            ax = fig.add_subplot()

            ax.bar(labels, good, label='good')
            ax.bar(labels, false, bottom=good, label='false')
            ax.bar(labels, miss, bottom=np.array(
                good) + np.array(false), label='miss')

            ax.set_ylabel('Amount')
            ax.set_title('Good False Miss')
            ax.legend()

            save_path = str(self.save_dir) + "/%s.png" % save_name
            ax.get_figure().savefig(save_path)
            self.logger.debug("Stacked Bar Plot Saved: %s" % save_path)
            plt.close()

            labels = [str(_) for _ in df.index.to_numpy()[:-1]]
            labels, good_rate, false_rate, miss_rate = zip(
                *sorted(zip(labels, good_rate, false_rate, miss_rate), key=lambda _: _[1]))

            fig = plt.figure(figsize=(30, 20))
            ax = fig.add_subplot()

            ax.bar(labels, good_rate, label='good')
            ax.bar(labels, false_rate, bottom=good_rate, label='false')
            ax.bar(labels, miss_rate, bottom=np.array(
                good_rate) + np.array(false_rate), label='miss')

            ax.set_ylabel('Rate')
            ax.set_title('Good False Miss')
            ax.legend()

            save_path = str(self.save_dir) + "/%s_rate.png" % save_name
            ax.get_figure().savefig(save_path)
            self.logger.debug("Stacked Bar Plot Saved: %s" % save_path)
            plt.close()

    def plot_pie(self, data_dict=None, data_list=None, ingredients=None, title=None) -> None:
        if data_dict is not None:
            ingredients = list(data_dict.keys())
            data_list = [data_dict[_] for _ in ingredients]
        # fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))
        fig, ax = plt.subplots(figsize=(12, 5))

        def func(pct, allvals):
            absolute = int(pct/100.*np.sum(allvals))
            return "{:.1f}%\n({:d})".format(pct, absolute)

        wedges, texts, autotexts = ax.pie(data_list, autopct=lambda pct: func(pct, data_list),
                                          textprops=dict(color="w"))

        ax.legend(wedges, ingredients,
                  title="error_type",
                  loc="best",
                  bbox_to_anchor=(1, 0, 0.5, 1))

        plt.setp(autotexts, size=8, weight="bold")

        ax.set_title(title)
        # plt.savefig(self.path, bbox_inches='tight', dpi=low_dpi)
        save_path = self.save_dir / (title + '.png')
        plt.savefig(save_path)
        self.logger.debug("Pie Saved: %s" % save_path)

        # plt.show()
        # ax.axis('equal')
        # plt.title(model_name, fontdict={'fontsize': 60, 'fontweight': 'bold'})
        # pie_path = os.path.join(tmp_dir, '{}_{}_pie.png'.format(model_name, rec_type))
        # plt.savefig(pie_path, bbox_inches='tight', dpi=low_dpi)
        # plt.close()
    def crop_img(self, img_path: str, bbox: list, save_path=None) -> None:
        """
        Description: save cropped images
        Param:  bbox: x,y,w,h
        Returns: 
        """
        img = Image.open('/' + img_path)
        cropped = img.crop(
            (bbox[0], bbox[1], bbox[0] + abs(bbox[2]), bbox[1] + abs(bbox[3])))
        if save_path is not None:
            shutil.os.makedirs(Path(save_path).parent, exist_ok=True)
            cropped.save(save_path)

    def draw_bbox(self, img_path: str, bbox_list: list, save_path=None, text_list=None) -> None:
        """
        Description: save cropped images
        Param:  bbox: x,y,w,h
        Returns: 
        """
        img = Image.open('/' + img_path)
        img_draw = ImageDraw.Draw(img)
        # ft = ImageFont.truetype(font=None, size=20)

        font = ImageFont.truetype("/share/analysis/fonts/DejaVuSans.ttf", size=np.floor(3e-2 * 1080 + 0.5).astype('int32'))
        # font = ImageFont.load_default()
        for idx, bbox in enumerate(bbox_list):
            color_map = {0: (255, 0, 0), 1: (0, 255, 0)}
            pos_map = {0: (bbox[0], bbox[1] - 30),
                       1: (bbox[0], bbox[1] + abs(bbox[3]))}
            # label_size = img_draw.textsize(text, font)
            text_origin = np.array([bbox[0], bbox[1]])
            # 在边界框的两点（左上角、右下角）画矩形，无填充，边框红色，边框像素为5
            img_draw.rectangle((bbox[0], bbox[1], bbox[0] + abs(bbox[2]), bbox[1] + abs(
                bbox[3])), fill=None, outline=color_map[idx], width=2)
            # if text is not None:
            # img_draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],fill=225)
            # img_draw.text(text_origin, text, fill=(0, 0, 0), font=font)

            img_draw.text(pos_map[idx], text_list[idx],
                          fill=color_map[idx], font=font)
        if save_path is not None:
            shutil.os.makedirs(Path(save_path).parent, exist_ok=True)
            img.save(save_path)
    
    def draw_bbox_0(self, obs_list: List[Obstacle] = [], save_path=None) -> None:
        """
        Description: save cropped images
        Param:  bbox: x,y,w,h
        Returns: 
        """
        if obs_list == 0:
            self.logger.warning("List[Obstacle] == 0, cannot draw bbox")
            return
        img_path = obs_list[0].img_path
        img = Image.open('/' + img_path)
        img_draw = ImageDraw.Draw(img)
        # ft = ImageFont.truetype(font=None, size=20)
        color_map = {'dt': (255, 0, 0), 'gt': (0, 255, 0), 'good':(125,125,125), 'train':(125,125,125)}
        font = ImageFont.truetype("/share/analysis/fonts/DejaVuSans.ttf", size=np.floor(3e-2 * 1080 + 0.5).astype('int32'))
        # font = ImageFont.load_default()
        for obs in obs_list:
            dtgt, class_name, bbox = obs.dtgt, obs.class_name, obs.bbox
            pos_map = {'gt': (bbox[0], bbox[1] - 30),
                       'dt': (bbox[0], bbox[1] + abs(bbox[3])),
                       'train': (bbox[0], bbox[1] - 30)}
            # label_size = img_draw.textsize(text, font)
            text_origin = np.array([bbox[0], bbox[1]])
            # 在边界框的两点（左上角、右下角）画矩形，无填充，边框红色，边框像素为5
            img_draw.rectangle((bbox[0], bbox[1], bbox[0] + abs(bbox[2]), bbox[1] + abs(
                bbox[3])), fill=None, outline=color_map[dtgt], width=2)
            # if text is not None:
            # img_draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],fill=225)
            # img_draw.text(text_origin, text, fill=(0, 0, 0), font=font)
            text = dtgt + '_' + class_name
            img_draw.text(pos_map[dtgt], text, fill=color_map[dtgt], font=font)
        if save_path is not None:
            shutil.os.makedirs(Path(save_path).parent, exist_ok=True)
            img.save(save_path)

    def plot_bar(self, data_dic, title):
        data_type_dic = {'train': [], 'test': [], 'bad_case': []}
        labels = list(data_dic['test'].keys())
        for label in labels:
            for data_type in data_type_dic.keys():
                if label not in data_dic[data_type]:
                    data_type_dic[data_type].append(0)
                    continue
                # print(data_dic)
                data_type_dic[data_type].append(data_dic[data_type][label])

        x = np.arange(len(labels))  # the label locations
        width = 0.2  # the width of the bars

        fig, ax = plt.subplots(figsize=(8, 5))
        # print(data_type_dic)
        rects1 = ax.bar(
            x - width, data_type_dic['train'], width/1.2, label='train')
        rects2 = ax.bar(x, data_type_dic['test'], width/1.2, label='test')
        rects3 = ax.bar(
            x + width, data_type_dic['bad_case'], width/1.2, label='bad_case')

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        # autolabel(rects1)
        # autolabel(rects2)
        # autolabel(rects3)

        # fig.tight_layout()

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Frequency')
        ax.set_title('%s analysis' % title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        save_path = self.save_dir / (title + '.png')
        fig.savefig(save_path)
        self.logger.debug("%s Chart Saved: %s" %
                         (title.capitalize(), save_path))

    def plot_confusion_matrix(self, gt_label, pred_label, title):
        if (len(gt_label) == 0) or (len(pred_label) != len(pred_label)):
            self.logger.warning("plot confusion matrix failed! gt: %d, dt: %d" \
                                % (len(gt_label), len(pred_label)))
            return
        name_set = set(gt_label)
        name_list = []
        vehicle_list = ['bus', 'truck', 'car', ]
        vru_list = ['tricycle', 'bicycle', 'rider', 'pedestrian']
        for obj in vehicle_list+vru_list:
            if obj in name_set:
                name_list.append(obj)
        if len(name_list) == 0:
            name_list = list(name_set)
        conf_matrix = confusion_matrix(gt_label, pred_label, labels=name_list)
        df_cm = pd.DataFrame(conf_matrix, index=name_list,
                             columns=name_list)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot()
        ax.set_title(title + ' class confusion matrix')

        sns.set(font_scale=1)
        ax = sns.heatmap(df_cm, annot=True, fmt='g')
        ax.set_xlabel('detection')
        ax.set_ylabel('ground_truth')
        save_path = self.save_dir / (title + '_confusion_matrix.png')
        fig.savefig(save_path)
        self.logger.debug("Confusion Matrix Heatmap Saved: %s" % save_path)

    def plot_kernel_density(self, data_list, label_list, save_path=None):
        """
        Summary
        -------
            Plot the kernerl density 

        Parameters
        ----------
            data_list: List[List[num]]
                list of list of float or int
            label_list: List[str]
                The label of each list
            save_path: str
                The path to save the plot
        """

        f, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 6))
        color_list = sns.color_palette('Set1', len(label_list))
        for i in range(len(label_list)):
            f = sns.kdeplot(
                data_list[i], shade=True, color=color_list[i], label=label_list[i], ax=ax)

        plt.tick_params(labelsize=12)
        plt.xlabel("Score", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(loc='best', prop={'size': 14})
        save_path = self.save_dir / 'kernel_density.png' if save_path is None else save_path
        plt.savefig(save_path)
        self.logger.debug("Kernel Density Saved: %s" % save_path)
    
    def plot_bird_view(self, obs_list: List[Obstacle] = None, save_path:str = None):
        if obs_list is None:
            obs_list = []
        topView = TopViewer(x_range=100, y_range=60, view_unit=0.05)
        topView.initTopView()
        color_dict = {'gt': (0,255,0), 'dt': (0, 0, 255)}

        for obs in obs_list:
            position = [obs.x, obs.y, obs.z]
            dimension = [obs.height, obs.length, obs.width]
            yaw = obs.yaw
            if pd.isna(yaw):
                continue
            # print(position, dimension, yaw, obs.dtgt)
            gtdt = obs.dtgt if obs.dtgt != None else 'gt'
            topView.drawVehicleObject(position, dimension, yaw, color=color_dict[gtdt],thickness=2)
        topView.drawVehicleObject([0,0,0],[0,40,40],0,color=(0,255,0),thickness=2,orient=False)
        if save_path is not None:
            cv2.imwrite(save_path, topView.view_mat)
