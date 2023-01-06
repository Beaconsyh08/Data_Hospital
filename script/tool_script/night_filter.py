import pandas as pd
from tqdm import tqdm
from datetime import time
import random

ROOT = "/root/data_hospital_data/front_night/dataframes/"
DF_LST = ["dataframe.pkl",]

night_start = 21
night_end = 4
obj_threshold = 5
amount = 1000

for pkl in DF_LST:
    # name = pkl.split("_")[0]
    # print(name)
    # name = "darker"
    
    df = pd.read_pickle(ROOT + pkl)["df"]
    df_jsons = list(set(df.json_path.to_list()))
    print(len(df_jsons))
    
    night_df = df[(df["time"] >= time(night_start, 0, 0)) | (df["time"] <= time(night_end, 0, 0))]
    night_df_jsons = list(set(night_df.json_path.to_list()))
    print(len(night_df_jsons))
    
    obj_df = night_df[night_df.objects_amount < obj_threshold]
    obj_df_jsons = list(set(obj_df.json_path.to_list()))
    print(len(obj_df_jsons))
    
    obj_df_jsons = random.sample(obj_df_jsons, amount)
    print(len(obj_df_jsons))
    
    # save_path = "/data_path/%s_night.txt" % name
    save_path = "/data_path/%d_%d_%d_night_bg.txt" % (night_start, night_end, obj_threshold)
    
    print(save_path)
    with open(save_path, "w") as output_file:
        for json_path in tqdm(obj_df_jsons):
            output_file.writelines(json_path + "\n")
            
            
