import os
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time

def load(json_path,count):
    with open(json_path.strip(),'r') as f:
        # print(count)
        pass

def load_and_dowmload_imgs(json_path,save_dir):
    try:
        with open(json_path.strip(),'r') as f:
            data_info = json.load(f)
        
        imgUrl = '/' + data_info['imgUrl_fake']
        save_path = os.path.join(save_dir, os.path.basename(imgUrl))
        os.system('cp {} {}'.format(imgUrl, save_path))
    except:
        print(json_path)

def load_and_filter_json(json_path,save_dir):
    with open(json_path.strip(),'r') as f:
        data_info = json.load(f)
    
    date_time = datetime.fromtimestamp(int(str(data_info['timestamp'])[:10]))
    timestamp_date = date_time.date()
    timestamp_time = date_time.time()
    night_tag = (timestamp_time < time(5, 0, 0)) | (timestamp_time > time(21, 0, 0))

jsontxt_path = '/share/2d-od/lei/docker_related/clip_test/clip10/clip_7.txt'
save_dir = '/share/2d-od/lei/docker_related/clip_test/clip10/clip_7'

if not os.path.exists(save_dir):
    os.system('mkdir -p {}'.format(save_dir))

with open(jsontxt_path,'r') as f:
    jsons = f.readlines()

jsons = tqdm(jsons)

count = 0
with ThreadPoolExecutor(max_workers=64) as pool:
    temp_infors = [pool.submit(load_and_dowmload_imgs, json_path, save_dir) for json_path in tqdm(jsons)]
    temp_infors = tqdm(temp_infors)
    data_infos = [t.result() for t in temp_infors]
    # for json_path in jsons:
    #     pool.submit(load,json_path,count)
    #     count += 1
