# train_path = "/data_path/v40_train.txt"
from lib2to3.pgen2.token import NAME
from multiprocessing.pool import ThreadPool
from os import dup
from tqdm import tqdm
import json
from src.utils.logger import get_logger
import pandas as pd
logger = get_logger()
import pickle
from collections import Counter
import os


def load_img_or_json(load_path: str) -> dict:
    json_paths = [_.strip() for _ in list(open(load_path, "r"))]
    res_dict_json = dict()
    res_dict_img = dict()
    
    def worker(_):
        json_path = _.strip()
        
        with open(json_path) as json_obj:
            json_info = json.load(json_obj)
            camera_orientation = json_info.get("camera_orientation")
            rel_infos = json_info.get("relative_sensors_data")
            json_ori = ""
            for rel_info in rel_infos:
                if rel_info.get("camera_orientation") == camera_orientation:
                    json_ori = rel_info["image_json"]
                    break
            
            if json_ori.startswith("oss:"):
                res_dict_json[json_path] = "/" + json_ori
            else:
                img_url = json_info["imgUrl"]
                res_dict_img[json_path] = img_url


    with ThreadPool(processes = 40) as pool:
        list(tqdm(pool.imap(worker, json_paths), total=len(json_paths), desc='ImgUrl Loading'))
        pool.terminate()
        
    return res_dict_json, res_dict_img


def load_img_url(load_path: str) -> dict:
    json_paths = [_.strip() for _ in list(open(load_path, "r"))]
    res_dict = dict()
    
    def worker(_):
        json_path = _.strip()
        
        with open(json_path) as json_obj:
            json_info = json.load(json_obj)
            img_url = json_info["imgUrl"]
            res_dict[json_path] = img_url


    with ThreadPool(processes = 40) as pool:
        list(tqdm(pool.imap(worker, json_paths), total=len(json_paths), desc='ImgUrl Loading'))
        pool.terminate()
        
    return res_dict


def save_to_pickle(pickle_obj: dict, save_path: str) -> None:
    with open(save_path, "wb") as pickle_file: 
        pickle.dump(pickle_obj, pickle_file)


if __name__ == '__main__':
    NAME = "0719"
    PATH = "/root/data_hospital/imgurl"

    all_train_path = "/data_path/v50_train.txt"
    save_path = "/root/data_hospital/miss_label_filter/%s" % NAME
    to_del_path = "%s/to_del.txt" % save_path
    
    all_train_pkl_path = "%s/all_trains.pkl" % PATH
    to_del_pkl_path = "%s/to_del.pkl" % PATH
    final_pkl_path = "%s/final.pkl" % PATH
    
    # to_del_dict_json, to_del_dict_img = load_img_or_json(to_del_path)
    # save_to_pickle(to_del_dict_json, "%s/to_del_dict_json.pkl" % PATH)
    # save_to_pickle(to_del_dict_img, "%s/to_del_dict_img.pkl" % PATH)
    
    # all_train_dict = load_img_url(all_train_path)
    # save_to_pickle(all_train_dict, all_train_pkl_path)
    
    all_train_dict = pd.read_pickle(all_train_pkl_path)
    to_del_dict_json = pd.read_pickle("%s/to_del_dict_json.pkl" % PATH)
    to_del_dict_img = pd.read_pickle("%s/to_del_dict_img.pkl" % PATH)
    
    to_del_json_lst = list(to_del_dict_json.values())
    to_del_img_lst = list(to_del_dict_img.values())
    
    print(len(to_del_json_lst))
    print(len(to_del_img_lst))
    
    keys = list(all_train_dict.keys())
    values = list(all_train_dict.values())

    print(len(keys))
    print(len(set(keys)))
    print(len(values))
    print(len(set(values)))
    
    dup_img = list((Counter(values)-Counter(list(set(values)))).elements())

    def worker(_):
        try:
            all_indexs = list(filter(lambda x: values[x] == _, range(len(values))))
            if len(all_indexs) > 1:
                print(len(all_indexs))
                for ind in all_indexs:
                    print(keys[ind])
                    
                    mod_time = os.path.getmtime(path)
                    
            # to_del_json = keys[values.index(_)]
            # to_del_img_json_lst.append(to_del_json)
        except:
            pass
    
    to_del_img_json_lst = []
        
    with ThreadPool(processes = 40) as pool:
        list(tqdm(pool.imap(worker, dup_img), total=len(dup_img), desc='ImgUrl Searching'))
        pool.terminate()
        
    # print(len(to_del_img_json_lst))
    # total_del_lst = to_del_json_lst + to_del_img_json_lst
    # print(len(total_del_lst))
    
    # with open("%s/to_del_train.txt" % PATH, "w") as output_file:
    #     for json_path in total_del_lst:
    #         output_file.writelines(json_path + "\n")
