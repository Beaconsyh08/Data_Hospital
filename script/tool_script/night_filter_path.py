import json
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from datetime import time
from datetime import datetime


def run(json_paths: list) -> None:
    night_lst, day_lst = [], []
    def worker(_):
        json_path = _.strip()
        with open(json_path) as json_obj:
            json_info = json.load(json_obj)
            try:
                timestamp = json_info["timestamp"] if type(json_info["timestamp"]) == int else int(json_path.split("/")[-1][:-5])
            except ValueError:
                timestamp = int(json_path.split("/")[-1][-21:-5])
            date_time = datetime.fromtimestamp(int(str(timestamp)[:10]))
            cur_time = date_time.time()
            
            if  time(6, 0, 0) < cur_time < time(19, 0, 0):
                day_lst.append(json_path)
            else:
                night_lst.append(json_path)

    with ThreadPool(processes = 40) as pool:
        list(tqdm(pool.imap(worker, json_paths), total=len(json_paths), desc='Map Loading'))
        pool.terminate()
        
    return night_lst, day_lst
    
if __name__ == '__main__':
    txt_path = "/data_path/2.2.8.0.txt"
    
    with open(txt_path) as data_file:
        json_paths = [_.strip() for _ in data_file]
        
    night_lst, day_lst = run(json_paths=json_paths)
    
    with open("/data_path/exp_night.txt", "w") as night_file:
        for _ in night_lst:
            night_file.writelines(_ + "\n")
            
    with open("/data_path/exp_day.txt", "w") as day_file:
        for _ in day_lst:
            day_file.writelines(_ + "\n")
    
        
    
