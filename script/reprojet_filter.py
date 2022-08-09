import pandas as pd 
from tqdm import tqdm

ROOT_PATH = "/root/data_hospital/0728v60/qa_cmh"
# BINS = [0, 0.1, 0.2, 0.5, 0.55, 0.6, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.85, 0.90, 1]
threshold_percent = 0.3
threshold = 0.6

df = pd.read_excel("%s/reproject_doctor/result.xlsx" % ROOT_PATH)
df["Mean Iou"].hist(bins=50).get_figure().savefig("%s/reproject_doctor/result_hist.png" % ROOT_PATH)

df.sort_values(by="Mean Iou", inplace=True, ascending=False)

# df_select = df.iloc[:int(threshold_percent * len(df))]
df_select = df[df["Mean Iou"] > threshold]

print(df_select)
print("Last Mean Iou: %f" % df_select.iloc[-1]["Mean Iou"])

count = 0
with open("%s/reproject_doctor/selected.txt" % ROOT_PATH, "w") as output_file:
    for file in tqdm(df_select.Txt):
        with open(file) as select_file:
            lines = [_.strip() for _ in select_file]
            count += len(lines)   
            for line in lines:
                output_file.writelines(line + "\n")

print("Number of Selected: %d" % count)

    
    
# print("Number of Selected: %d" % df_select["Count"].sum())
# print("Total: %d" % df["Count"].sum())
# print("Percentage: %f" % df_select["Count"].sum() / df["Count"].sum())

