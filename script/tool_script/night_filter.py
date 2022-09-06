from logging import RootLogger
import pandas as pd
from datetime import time

ROOT = "/share/analysis/result/eda/dataframes/"
DF_LST = ["lucas_beibao.pkl",]
for pkl in DF_LST:
    name = pkl.split("_")[0]
    print(name)
    
    df = pd.read_pickle(ROOT + pkl)["df"]
    df_jsons = list(set(df.json_path.to_list()))
    print(len(df_jsons))
    
    night_df = df[(df["time"] > time(19, 0, 0)) & (df["time"] < time(6, 0, 0))]
    night_df_jsons = list(set(night_df.json_path.to_list()))
    print(len(night_df_jsons))
    
    with open("/data_path/%s_night.txt" % name, "w") as output_file:
        for json_path in night_df_jsons:
            output_file.writelines(json_path + "\n")