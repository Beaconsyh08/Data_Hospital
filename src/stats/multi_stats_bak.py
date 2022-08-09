from pathlib import Path
from src.stats.stats import Stats
import pandas as pd
import src.data_manager.data_manager as dm
import os
import collections
from PIL import Image
import shutil
import pickle
from tqdm import tqdm
import time
import threading
from configs.config import StatsConfig
from src.data_manager.data_manager_creator import data_manager_creator
from configs.config import ClusterConfig, Config, EvalDataFrameConfig,\
                     VisualizationConfig, RetrievalDataFrameConfig, TrainDataFrameConfig
from src.utils.file_io import crop_img
from src.utils.logger import get_logger
from src.visualization.visualization import Visualizer


class MultiStats(Stats):
    def __init__(self, df: pd.DataFrame = None, df_path: str = None, cfg: dict = None) -> None:
        super(MultiStats, self).__init__(df=df, df_path=df_path, cfg=cfg)
        self.logger = get_logger()
        self.visualizer = Visualizer(VisualizationConfig)


    def comparer(self, one_df: pd.DataFrame, another_df: pd.DataFrame) -> pd.DataFrame:
        # TODO Not finished; think about what to compared, and how the output should looks like
        """
        Summary
        -------
            Comapare and present the differnt rows between two DataFrame

        Parameters
        ----------
            one_df: pd.DataFrame
                [description]
            another_df: pd.DataFrame
                 [description]
            subset: list
                [description]

        Returns
        -------
            pd.DataFrame: [description]
        """
        # compare two dataframe, return rows in different
        return pd.concat([one_df, another_df]).duplicated(keep=False)


    def cluster_metric_comparer(self) -> None:
        root_path = StatsConfig.ROOT + "/dataframes/"
        pk_lst = []
        for each in os.listdir(root_path):
            if len(each.split("_")) > 2:
                pk_lst.append(dm.load_from_pickle(root_path + "/" + each, "info"))

        pd.options.display.float_format = '{:,.2f}'.format
        res_df = pd.DataFrame(pk_lst).set_index("method")
        return res_df

    def case_transfer(self, df1, df2):
        g2g = (df1['flag'] == 'good') & (df2['flag'] == 'good')
        g2b = (df1['flag'] == 'good') & (df2['flag'] == 'miss')
        b2g = (df1['flag'] == 'miss') & (df2['flag'] == 'good')
        b2b = (df1['flag'] == 'miss') & (df2['flag'] == 'miss')


        total = sum((df1['flag'] == 'good') |(df1['flag'] == 'miss'))
        self.logger.info('g2g: %d, g2b: %d, b2g: %d, b2b: %d', sum(g2g), sum(g2b), sum(b2g), sum(b2b))
        self.logger.info('g2g: %.4f, g2b: %.4f, b2g: %.4f, b2b: %.4f', sum(g2g)/total, sum(g2b)/total, sum(b2g)/total, sum(b2b)/total)
        self.save_img(df1.loc[g2b[g2b].index], 'g2b')
        self.save_img(df1.loc[b2g[b2g].index], 'b2g')
    
    def save_img(self, df, tag):
        cnt = 0
        for index, row in df.iterrows():
            img_path = index.split("@")[0]
            bbox = [row['bbox_x'],row['bbox_y'],row['bbox_w'],row['bbox_h']]
            save_path = os.path.join('images', tag, str(cnt) + '.jpg')
            crop_img(img_path, bbox, save_path)
            cnt += 1
    
    def cluster_compare(self, df1, df2):
        df = df1.copy(deep=True)
        df = df[df['flag'] != 'false']
        df2 = df2[df2['flag'] != 'false']
        for index, row in tqdm(df2.iterrows(), total=len(df2)):
            if index not in df.index:
                continue
            df.loc[index, 'flag'] = df.loc[index, 'flag'] + '_' + df2.loc[index, 'flag']
        cluster_id_list = df['cluster_id'].tolist()
        cluster_stats = {}
        for id in tqdm(cluster_id_list):
            d = df[df['cluster_id'] == id]['flag'].value_counts().to_dict()
            cluster_stats[id] = d
        pkl_file = open('cluster_stats.pkl', 'wb')
        pickle.dump(cluster_stats, pkl_file)
        print(cluster_stats)
    
    def compare_multi_version(self, df_multi_version: list, show=False):
        self.logger.info("Start compare multi version")
        self.qa_base_df = df_multi_version[0]
        df_base = df_multi_version[0]
        dic_multi = collections.OrderedDict(Index=[], flag_base=[])

        dic_others_version = [df.to_dict() for df in df_multi_version[1:]]
        for row in tqdm(df_base.itertuples(), total=len(df_base)):
            t0 = time.time()
            index,flag = getattr(row, 'Index'), getattr(row, 'flag')
            t1 = time.time()
            if (flag != 'good') and (flag != 'miss'):
                continue
            dic_multi['Index'].append(index)
            dic_multi['flag_base'].append(flag)
            
            for idx, dic_version in enumerate(dic_others_version):
                column_name = 'flag_' + str(idx+1)
                
                if column_name not in dic_multi:
                    dic_multi[column_name] = []
                # t2 = time.time()
                dic_multi[column_name].append(dic_version['flag'][index])
                # dic_multi[column_name].append(df.loc[index,'flag'])
                # t3 = time.time()
                # self.logger.info("t3-t2: %.5f", t3-t2)
        df_multi_comp = pd.DataFrame.from_dict(dic_multi)
        df_multi_comp.set_index('Index',  inplace=True) 
        self._define_case_change(df_multi_comp)
        self.qa_base_df['multi_version_info'] = None
        for row in tqdm(df_multi_comp.itertuples(), total=len(df_multi_comp)):
            index, multi_version =  getattr(row, 'Index'), getattr(row, 'multi_version')
            try:
                self.qa_base_df.at[index, 'multi_version_info'] = multi_version
            except:
                continue
        temp_dm = dm.DataManager(self.qa_base_df)
        temp_dm.save_to_pickle(Config.DATAFRAME_PATH)
        print(df_multi_comp)
        if show:
            for row in tqdm(df_multi_comp.itertuples(), total=len(df_multi_comp)):
                index, multi_version =  getattr(row, 'Index'), getattr(row, 'multi_version')
                if multi_version == 'easy':
                    continue
                if df_base.at[index, 'kind'] != 'P0':
                    continue
                img_path = index.split('@')[0]
                img_name = Path(img_path).stem
                bbox_list = [[df_base.at[index, 'bbox_x'], df_base.at[index, 'bbox_y'],
                              df_base.at[index, 'bbox_w'],df_base.at[index, 'bbox_h']]]
                flag, class_name, id = df_base.at[index, 'flag'],\
                                       df_base.at[index, 'class_name'],\
                                       df_base.at[index, 'id']

                save_path = os.path.join('./multi_version/', multi_version, \
                           '_'.join([flag, class_name, img_name, str(int(id))]) + '.jpg')
                class_name_list = [class_name]
                self.visualizer.draw_bbox(img_path, bbox_list, save_path, text_list = class_name_list)
                t = threading.Thread(target=self.visualizer.draw_bbox, args=(img_path, bbox_list, save_path, class_name_list))
                t.start()
        self.logger.info("Analyze multi version finished")
    
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
    
    def anlysis_retrieval(self, show=False):
        try:
            retrieval_data = data_manager_creator(RetrievalDataFrameConfig)
            retrieval_data.load_from_json()
            retrieval_dataframe = retrieval_data.getter()
        except:
            retrieval_dataframe = retrieval_data.getter()
            self.logger.warning("load retrieval data error !")

        train_data = data_manager_creator(TrainDataFrameConfig)
        train_data.load_from_json()
        train_dataframe = train_data.getter()
        def get_train_info(retrieval_info, imgurl_train_list):
            train_info = []
            imgurl_train_list=set(imgurl_train_list)
            for info in retrieval_info:
                imgUrl = info['imgUrl']
                img_name = imgUrl.split('/')[-1].split('.')[0]
                if img_name in imgurl_train_list:
                    train_info.append(info)
            return train_info

        def get_retrieval_info(df_retrieval):
            res_info = []
            for row in df_retrieval.itertuples():
                bbox = [row.value_bbox_x, row.value_bbox_y, \
                        row.value_bbox_w, row.value_bbox_h]
                imgUrl, id, class_name = row.value_img_url, row.value_id, row.value_class_name
                
                info = dict(imgUrl=imgUrl, id=id, class_name=class_name, bbox=bbox)
                res_info.append(info)
            return res_info
        retrieval_results = []
        train_results = []
        # imgurl_train_list = [index.split('@')[0][1:] for index in train_dataframe.index]
        imgurl_train_list = [_.split('/')[-1].split('.')[0] for _ in train_dataframe.index]
        for row in tqdm(self.qa_base_df.itertuples(), total=len(self.qa_base_df)):
            index, id, flag, priority = getattr(row, 'Index'), getattr(row, 'id'), \
                                        getattr(row, 'flag'), getattr(row, 'priority')
            if priority != 'P0':
                retrieval_results.append(None)
                train_results.append(None)
                continue
            img_path = index.split('@')[0][1:]
            res = retrieval_dataframe[(retrieval_dataframe['key_img_url'] == img_path) \
                            & (retrieval_dataframe['key_obj_id'] == id)]
            if len(res) == 0:
                retrieval_results.append(None)
                train_results.append(None)
                continue
            retrieval_info = get_retrieval_info(res)
            retrieval_results.append(retrieval_info)

            train_info = get_train_info(retrieval_info, imgurl_train_list)
            train_results.append(train_info)

        self.qa_base_df['retrieval_results'] = retrieval_results
        self.qa_base_df['train_results'] = train_results
        temp_dm = dm.DataManager(self.qa_base_df)
        temp_dm.save_to_pickle(Config.DATAFRAME_PATH)
        if show:
            cnt = 0
            for row in tqdm(self.qa_base_df.itertuples(), total=len(self.qa_base_df)):
                priority, flag = getattr(row, 'priority'),getattr(row, 'flag')
                if priority != 'P0' or flag != 'miss':
                    continue
                index, class_name = getattr(row, 'Index'), getattr(row, 'class_name')
                bbox = [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h]
                img_path = index.split('@')[0]
                cnt += 1
                case_change = getattr(row, 'multi_version_info')
                save_dir = os.path.join('retrieval_results', case_change, class_name, str(cnt))
                self.visualizer.crop_img(img_path, bbox, save_dir + '/0.jpg')
                
                train_results = getattr(row, 'train_results')
                if train_results is None:
                    continue

                for idx, res in enumerate(train_results):
                    imgUrl, bbox = res['imgUrl'], res['bbox']
                    # print(imgUrl)
                    save_path = os.path.join(save_dir, str(idx+1)+'.jpg')
                    t = threading.Thread(target=self.visualizer.crop_img, args=(imgUrl, bbox, save_path))
                    t.start()


if __name__ == "__main__":
    ms = MultiStats()
    # df_1 = dm.load_from_pickle('/share/analysis/result/dataframes/eval_dataframe_1.pkl')
    # df_2 = dm.load_from_pickle('/share/analysis/result/dataframes/eval_dataframe_2.pkl')
    # # print(df_1)
    # # print(df_2)
    # # df_3 = dm.merge_dataframe_rows(df_1, df_2).getter()
    # # print(df_3[df_3.duplicated(subset=["dtgt", "flag"])])
    # print(ms.cluster_metric_comparer())
    multi_eval_path = [
        '/data_path/v3_eval.txt',
        '/data_path/v4_eval.txt',
    ]
    df_multi_version = []
    for path in multi_eval_path:
        EvalDataFrameConfig.JSON_PATH = path
        temp = data_manager_creator(EvalDataFrameConfig)
        temp.load_from_json()
        df_multi_version.append(temp.df.copy())
    
    ms.compare_multi_version(df_multi_version, show=False)
    # ms.anlysis_retrieval()
    # ms.case_transfer(v0, v1.getter())
    # ms.cluster_compare(v0, v1.getter())