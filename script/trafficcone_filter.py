import pandas as pd

df_path = "/root/data_hospital_data/0728v60/test0809/dataframes/reproject_dataframe.pkl"
df = pd.read_pickle(df_path)["df"]
print(df.groupby("yaw").size())
selected_df = df[df.class_name.isin(["trafficCone", "traffic-cone"])]
json_path = set(selected_df.json_path.to_list())
with open("/data_path/traffic_cone.txt", "w") as output_file:
    for json_p in json_path:
        output_file.writelines(json_p[1:] + "\n")