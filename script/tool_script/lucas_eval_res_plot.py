import json
import pandas as pd
import matplotlib.pyplot as plt
import os 

plt.rc('font', size=14)
plt.rc('axes', titlesize=24) 
plt.rc('axes', labelsize=20)

MODELS = ["BASE20",  "BASE20+RN2", "BASE20+FN2", "BASE20+FN4", "BASE20+FN6", "BASE20+FN8", "BASE20+FN10"]
TEST_SETS = ["day_test", "night_test"]
# TEST_SETS = ["day_test"]
NAME = "FN_COMPARE"
MODE = "analysis"
SAVE_ROOT = "/share/analysis/syh/vis/gan"
os.makedirs(SAVE_ROOT, exist_ok=True)

for test_set in TEST_SETS:
    res_df = pd.DataFrame()
    for model in MODELS:
        if MODE == "lucas_eval":
            json_path = "../data_hospital_data/%s/%s/evaluate_processor/result.json" % (model, test_set)
            with open(json_path) as json_obj:
                result_json = json.load(json_obj)
                res = result_json["all_P0_bev"]
                p, r, f = res["precision"], res["recall"], res["f1_score"]
                res_df.at[model, "precision"] = p
                res_df.at[model, "recall"] = r
                res_df.at[model, "f1"] = f
        elif MODE == "analysis":
            json_path = "../cases_analysis_data/%s/%s.json" % (test_set, model)
            try:
                with open(json_path) as json_obj:
                    result_json = json.load(json_obj)
                    res = result_json
                    p, r, f = res["P0_Precision"], res["P0_Recall"], res["P0_F1_Score"]
                    res_df.at[model, "precision"] = p
                    res_df.at[model, "recall"] = r
                    res_df.at[model, "f1"] = f
            except FileNotFoundError:
                print("%s Not Found" % model) 

    
    save_path = "%s/%s_%s.xlsx" % (SAVE_ROOT, NAME, test_set)
    print(save_path)
    res_df.to_excel(save_path)
    
    res_df.plot.bar(figsize=(15, 10), rot=25, title=test_set)
    save_path = "%s/%s_%s_bar.png" % (SAVE_ROOT, NAME, test_set)
    plt.savefig(save_path)
    print(save_path)
    
    res_df.plot.line(figsize=(15, 10), rot=25, title=test_set)
    save_path = "%s/%s_%s_line.png" % (SAVE_ROOT, NAME, test_set)
    plt.savefig(save_path)
    print(save_path)
