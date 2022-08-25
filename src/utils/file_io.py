#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   file_io.py
@Time    :   2021/11/29 15:19:52
@Author  :   chenminghua 
'''

import os
import json
from pathlib import Path
import shutil
from PIL import Image,ImageDraw
from tqdm import tqdm
from src.utils.logger import get_logger
logger = get_logger()

def save_query_files(output_query_dir, df_sampler):
    """
    Description: output query
    Param: 
    Returns: 
    """
    shutil.os.makedirs(output_query_dir, exist_ok=True)
    # df = load_from_pickle(SamplerConfig.DATAFRAME_PATH)
    # self.logger.info('\t'+ df.to_string().replace('\n', '\n\t')) 
    # print(load_from_pickle(SamplerConfig.DATAFRAME_PATH, "info"))
    # print(df)
    saved_file_name = set()
    for index, data in tqdm(df_sampler.iterrows(), total=len(df_sampler)):
        if not data['sampling']:
            continue
        json_path = Path(data['json_path'].strip())
        try:
            if json_path in saved_file_name:
                json_path = output_query_dir / json_path.name
                with open(str(json_path), 'r') as f:
                    json_map = json.load(f)
            else:
                with open(str(json_path), 'r') as f:
                    json_map = json.load(f)
                json_map['bad_cases'] = []
                saved_file_name.add(json_path)
        except:
            continue
        case = dict(
            result = 1,
            iou = 0,
            gt_info  = dict(id=data['id'], className=data['class_name'], 
                            bbox = [data['bbox_x'], data['bbox_y'], 
                            data['bbox_w'], data['bbox_h']]),
            dt_info = dict()   
        )
        json_map['bad_cases'].append(case)
        json_map['rtk_data'] = []
        json_map['miss_detection'] = []
        json_map['false_detection'] = []
        output_path = output_query_dir / json_path.name
        with open(str(output_path), 'w') as f:
            json.dump(json_map, f)
    logger.debug('The Query Files Have Been Created in: %s', output_query_dir)


def crop_img(img_path:str, bbox:list, save_path = None)->None:
    """
    Description: save cropped images
    Param:  bbox: x,y,w,h
    Returns: 
    """
    img = Image.open('/' + img_path)
    cropped = img.crop((bbox[0], bbox[1], bbox[0] + abs(bbox[2]), bbox[1] + abs(bbox[3])))
    shutil.os.makedirs(Path(save_path).parent, exist_ok=True)
    if save_path is not None:
        cropped.save(save_path)

def draw_bbox(img_path:str, bbox:list, save_path = None)->None:
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

def read_json(json_file):
    """
    load json file
    """
    json_map = None
    with open(json_file, 'r') as fp:
        try:
            json_map = json.load(fp)
        except:
            logger.error("failed to load: %s", json_file)
    return json_map

def write_json(json_file, json_map):
    """
    dump json file
    """
    shutil.os.makedirs(Path(json_file).parent, exist_ok=True)
    with open(json_file, 'w') as fp:
        fp.write(json.dumps(json_map ,indent=2))

def read_txt(txt_file):
    """
    load json file
    """
    contents = None
    if not os.path.exists(txt_file):
        logger.error('%s not exists!'%(txt_file))
        return
    with open(txt_file, 'r') as fp:
        contents = fp.readlines()
    return contents