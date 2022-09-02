import json
import pandas as pd

save_dir = "/share/analysis/result/qa_res/DashBoard V0.1 Data"
json_example = "%s/example.json" % save_dir

df = pd.read_excel('/share/analysis/result/qa_res/Version Data.xlsx')
print(df)

with open(json_example) as example:
    exp_json = json.load(example)

print(exp_json)

for row in df.itertuples():
    new_json = exp_json
    new_json["indicator_panel_json"]["P0_Precision"] = round(row.Precision / 100, 4)
    new_json["indicator_panel_json"]["P0_Recall"] = round(row.Recall / 100, 4)
    new_json["indicator_panel_json"]["P0_F1_Score"] = round(row.F1/100, 4)
    new_json["indicator_panel_json"]["Data_Amount"] = row.Data
    
    with open("%s/%s.json" % (save_dir, row.Version[1:]), "w") as output_file:
        json.dump(new_json, output_file)
