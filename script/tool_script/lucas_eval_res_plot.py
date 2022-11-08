import json
import pandas as pd
import matplotlib.pyplot as plt
import os 

plt.rc('font', size=14)
plt.rc('axes', titlesize=24) 
plt.rc('axes', labelsize=20)

MODELS = ["BASE20", "BASE20+FN2", "BASE20+FN4", "BASE20+FN6", "BASE20+FN8", "BASE20+FN10"]
TARGET = "BASE20+RN2"
TEST_SETS = ["day_test", "night_test"]
# TEST_SETS = ["day_test"]
NAME = "FN_COMPARE"
MODE = "analysis"
SAVE_ROOT = "/share/analysis/syh/vis/gan"
os.makedirs(SAVE_ROOT, exist_ok=True)

def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
            

for test_set in TEST_SETS:
    res_df = pd.DataFrame()
    json_path = "../cases_analysis_data/%s/%s.json" % (test_set, TARGET)
    with open(json_path) as json_obj:
        result_json = json.load(json_obj)
        res = result_json
        p, r, f = res["P0_Precision"], res["P0_Recall"], res["P0_F1_Score"]
        target = f
    
    for model in MODELS:
        if MODE == "lucas_eval":
            json_path = "../data_hospital_data/%s/%s/evaluate_processor/result.json" % (model, test_set)
            with open(json_path) as json_obj:
                result_json = json.load(json_obj)
                res = result_json["all_P0_bev"]
                p, r, f = res["precision"], res["recall"], res["f1_score"]
                res_df.at[model, "f1"] = f
                res_df.at[model, TARGET] = target
                # res_df.at[model, "precision"] = p
                # res_df.at[model, "recall"] = r
                
        elif MODE == "analysis":
            json_path = "../cases_analysis_data/%s/%s.json" % (test_set, model)
            with open(json_path) as json_obj:
                result_json = json.load(json_obj)
                res = result_json
                p, r, f = res["P0_Precision"], res["P0_Recall"], res["P0_F1_Score"]
                # res_df.at[model, "precision"] = p
                # res_df.at[model, "recall"] = r
                res_df.at[model, "f1"] = f
                res_df.at[model, TARGET] = target
                

    plt.rcParams["figure.figsize"] = [15, 10]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    save_path = "%s/%s_%s.xlsx" % (SAVE_ROOT, NAME, test_set)
    print(save_path)
    res_df.to_excel(save_path)
    
    res_df.f1.plot(kind="bar", color='yellowgreen', width=0.4)
    res_df.f1.plot(kind="line", marker='*', color='darkviolet', ms=10)
    res_df[TARGET].plot(kind="line", color='darkred')
    upper = max(res_df.f1.max(), res_df[TARGET].max()) + 0.02
    lower = min(res_df.f1.min(), res_df[TARGET].min()) - 0.02
    
    plt.ylim((lower, upper))
    plt.legend()
    plt.title("F1 Trend as Generated Night Data Increasing: %s" % test_set.capitalize())
    addlabels(MODELS, round(res_df.f1, 4))
    addlabels([MODELS[-1]], [round(res_df[TARGET].mean(), 4)])
    
    save_path = "%s/%s_%s.png" % (SAVE_ROOT, NAME, test_set)
    plt.savefig(save_path)
    print(save_path)
