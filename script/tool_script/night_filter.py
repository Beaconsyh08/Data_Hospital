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
    
    day_df = df[(df["time"] < time(19, 0, 0)) & (df["time"] > time(6, 0, 0))]
    day_df_jsons = list(set(day_df.json_path.to_list()))
    print(len(day_df_jsons))
    
    with open("/data_path/%s_day.txt" % name, "w") as output_file:
        for json_path in day_df_jsons:
            output_file.writelines(json_path + "\n")