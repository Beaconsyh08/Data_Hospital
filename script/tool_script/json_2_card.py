import json
import sys
import os
from tqdm import  tqdm
sys.path.append("/root/haomo_ai_framework")
from haomoai.cards import CardOperation
# from haomoai.cards import cardapi
from random import random
from pathlib import Path
from src.utils.file_io import write_json

oss_urls = []
json_path = '/share/analysis/syh/temp/dt' # json_files文件夹下保存了所有用于创建卡片的json文件，里面的json文件不在oss上，使用时修改此地址即可
card_inst = CardOperation()

for _, _, files in os.walk(json_path):
    for name in tqdm(files):
        js_name = json_path + '/' + name
        # print(js_name)
        with open(js_name, 'r', encoding='utf-8') as f:
            js_str = str(json.load(f))
            oss_urls.append(js_str)

card_id = card_inst.create_card_w_append( project='qa', media_name='sidecam_1205_dt', target_dir=json_path) # 设置生成卡片的 project 和 media_name
print('card_id: ', card_id)