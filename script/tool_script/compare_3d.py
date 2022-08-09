import json
import pandas as pd 
import os

SAVE_PATH = "/share/analysis/result/qa_res/clean156_weekly"
EXP = "/share/analysis/result/qa_res/clean156_weekly/qa_beijing_baoding_clean_result.json"
CONTROL = "/share/analysis/result/qa_res/clean156_weekly/qa_beijing_baoding_result.json"

COMBINED_LST = []

with open(CONTROL) as con_file:
    res_dict = dict()
    res_dict["name"] = "CONTROL"
    json_obj = json.load(con_file)
    res_vehicle = json_obj["0-20m"]["vehicle"]
    res_dict["Vehicle_pos"] = res_vehicle["position_error"]["mean"]
    res_dict["Vehicle_yaw"] = res_vehicle["yaw_error"]["mean"]
    res_dict["Vehicle_x"] = res_vehicle["pos_x_error"]["mean"]
    res_dict["Vehicle_y"] = res_vehicle["pos_y_error"]["mean"]
    
    res_vru = json_obj["0-20m"]["vru"]
    res_dict["VRU_pos"] = res_vru["position_error"]["mean"]
    res_dict["VRU_yaw"] = res_vru["yaw_error"]["mean"]
    res_dict["VRU_x"] = res_vru["pos_x_error"]["mean"]
    res_dict["VRU_y"] = res_vru["pos_y_error"]["mean"]
    COMBINED_LST.append(res_dict)
    
    
with open(EXP) as exp_file:
    res_dict = dict()
    res_dict["name"] = "EXP"
    json_obj = json.load(exp_file)
    res_vehicle = json_obj["0-20m"]["vehicle"]
    res_dict["Vehicle_pos"] = res_vehicle["position_error"]["mean"]
    res_dict["Vehicle_yaw"] = res_vehicle["yaw_error"]["mean"]
    res_dict["Vehicle_x"] = res_vehicle["pos_x_error"]["mean"]
    res_dict["Vehicle_y"] = res_vehicle["pos_y_error"]["mean"]
    
    res_vru = json_obj["0-20m"]["vru"]
    res_dict["VRU_pos"] = res_vru["position_error"]["mean"]
    res_dict["VRU_yaw"] = res_vru["yaw_error"]["mean"]
    res_dict["VRU_x"] = res_vru["pos_x_error"]["mean"]
    res_dict["VRU_y"] = res_vru["pos_y_error"]["mean"]
    COMBINED_LST.append(res_dict)
    
    
df = pd.DataFrame(COMBINED_LST)
df = df.set_index("name")
df = df.T
df["DIFF"] = df["CONTROL"] - df["EXP"]
df["PERCENT"] = df["DIFF"] / df["CONTROL"]
df.to_excel("%s/3d_result_2.xlsx" % SAVE_PATH)