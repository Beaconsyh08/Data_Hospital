import pandas as pd 


json_path = "/root/data-juicer/outputs/demo-gn6/demo-processed_all.jsonl"
output_path = "/mnt/share_disk/songyuhao/data/data_cleaning/v0.2.parquet"
df = pd.read_json(json_path, lines=True)
df["date"] = [_.split("_")[-1] for _ in df['carday_id']]
df_s = df[df["date"] >= "2023-08-01"]
df_s.to_parquet(output_path)
print(df_s)