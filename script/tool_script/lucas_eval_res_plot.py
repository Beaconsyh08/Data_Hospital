import json
import pandas as pd
import matplotlib.pyplot as plt
import os 

plt.rc('font', size=14)
plt.rc('axes', titlesize=24) 
plt.rc('axes', labelsize=20)

IS2D = False
ISSHARE = False
TARGET = "BASE20"
TEST_SETS = ["day_test", "night_test", "night_test_qa_frame"]
NAME = "BASE20_HM"
MODE = "analysis"
# MODELS = ["BASE20", "BASE20+FN2", "BASE20+FN4", "BASE20+FN6", "BASE20+FN8", "BASE20+FN10"]
# MODELS = ["BASE20", "BASE20+FN2", "C2_BASE20+FN2", "STROTSS-BASE20+FK2", "BASE20+FN10", "C2_BASE20+FN10", "BASE20+RN2+FN2", "C2_BASE20+RN2+FN2", "STROTSS-BASE20+FK2+RN2"]
# MODELS = ["BASE20", "BASE20+FN2", "C2_BASE20+FN2", "STROTSS-BASE20+FK2", "BASE20+FN6", "C2_BASE20+FN6", "STROTSS-BASE20+FK6w", "BASE20+FN10", "C2_BASE20+FN10", "STROTSS-BASE20+FK10w", "BASE20+RN2+FN2", "C2_BASE20+RN2+FN2", "STROTSS-BASE20+FK2+RN2"]
# MODELS = ["BASE20", "BASE20+FN10", "C2_BASE20+FN10", "STROTSS-BASE20+FK10w"]
# MODELS = ["BASE20", "BASE20+FN10", "C2_BASE20+FN10", "STROTSS-BASE20+FK10w"]
# MODELS = ["CARBASE", "CARBASE+FN10", "CARBASE+FN50"]
MODELS = ["BASE20", "C2_BASE20+FN2", "C2_BASE20+FN6", "C2_BASE20+FN10", "C2_BASE20+FN7"]
# MODELS = ["BASE20", "BASE20+FN2", "BASE20+FN6", "BASE20+FN10", "C2_BASE20+FN7"]





MODELS = [_ + "_2d" for _ in MODELS] if IS2D else MODELS
ROOT = "/share/analysis/syh/eval" if ISSHARE else "/root/cases_analysis_data"

SAVE_ROOT = "/share/analysis/syh/vis/gan"
os.makedirs(SAVE_ROOT, exist_ok=True)

def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
            

for test_set in TEST_SETS:
    res_df = pd.DataFrame()
    json_path = "%s/%s/%s_2d.json" % (ROOT, test_set, TARGET) if IS2D else "%s/%s/%s.json" % (ROOT, test_set, TARGET)
    not_found = []    
    with open(json_path) as json_obj:
        result_json = json.load(json_obj)
        res = result_json
        p, r, f = res["P0_Precision"], res["P0_Recall"], res["P0_F1_Score"]
        target = f
    
    for model in MODELS:
        print(model)
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
            # json_path = "/share/analysis/syh/eval/%s/%s.json" % (test_set, model)
            json_path = "%s/%s/%s.json" % (ROOT, test_set, model)
            try:
                with open(json_path) as json_obj:
                    result_json = json.load(json_obj)
                    res = result_json
                    p, r, f = res["P0_Precision"], res["P0_Recall"], res["P0_F1_Score"]
                    # res_df.at[model, "precision"] = p
                    # res_df.at[model, "recall"] = r
                    res_df.at[model, "f1"] = f
                    res_df.at[model, TARGET] = target
            except FileNotFoundError:
                not_found.append(model)
                print("No Result Found: %s %s" % (test_set, model))
    
    EX_MODELS = [_ for _ in MODELS if _ not in not_found]

    plt.rcParams["figure.figsize"] = [15, 10]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    save_path = "%s/%s_%s.xlsx" % (SAVE_ROOT, NAME, test_set)
    print(save_path)
    res_df.to_excel(save_path)
    
    colors = []
    for model in EX_MODELS:
        if "STROTSS" in model:
            colors.append("darkorange")
        elif "C2" in model:
            colors.append("darkred")
        else:
            colors.append("yellowgreen")
    
    bar_chart = res_df.f1.plot(kind="bar", color=colors, width=0.4)
    res_df.f1.plot(kind="line", marker='*', color='darkviolet', ms=10)
    res_df[TARGET].plot(kind="line", color='darkred', rot = 20)
    # upper = max(res_df.f1.max(), res_df[TARGET].max()) + 0.015
    upper = max(res_df.f1.max(), res_df[TARGET].max()) * 1.01
    
    # lower = min(res_df.f1.min(), res_df[TARGET].min()) - 0.015
    lower = min(res_df.f1.min(), res_df[TARGET].min()) * 0.99
    
    plt.ylim((lower, upper))
    plt.legend()
    title_prefix = "2D " if IS2D else ""
    plt.title("%sF1 Trend as Generated Night Data Increasing: %s" % (title_prefix, test_set.capitalize()))
    addlabels(EX_MODELS, round(res_df.f1, 4))
    addlabels([EX_MODELS[-1]], [round(res_df[TARGET].mean(), 4)])
    
    save_path = "%s/%s_%s.png" % (SAVE_ROOT, NAME, test_set)
    plt.savefig(save_path)
    print(save_path)
    
    EX_MODELS = MODELS
