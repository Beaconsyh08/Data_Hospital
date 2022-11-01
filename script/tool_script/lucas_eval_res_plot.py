import json
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', size=16)
plt.rc('axes', titlesize=24) 
plt.rc('axes', labelsize=20)

MODELS = ["BASE20+RN2", "BASE20+FN2", "BASE20+FN4", "BASE20+FN6", "BASE20+FN8", "BASE20+FN10"]
TEST_SETS = ["day_test", "night_test"]
NAME = "FN_COMPARE"

for test_set in TEST_SETS:
    res_df = pd.DataFrame()
    for model in MODELS:
        json_path = "../data_hospital_data/%s/%s/evaluate_processor/result.json" % (model, test_set)
        with open(json_path) as json_obj:
            result_json = json.load(json_obj)
            res = result_json["all_P0_bev"]
            p, r, f = res["precision"], res["recall"], res["f1_score"]
            res_df.at[model, "precision"] = p
            res_df.at[model, "recall"] = r
            res_df.at[model, "f1"] = f
            
    save_path = "/share/analysis/syh/vis/%s_%s.xlsx" % (NAME, test_set)
    print(save_path)
    res_df.to_excel(save_path)
    res_df.plot.bar(figsize=(15, 10), rot=0, title=test_set)
    save_path = "/share/analysis/syh/vis/%s_%s.png" % (NAME, test_set)
    plt.savefig(save_path)
    print(save_path)
