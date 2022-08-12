import json

PATH = "/share/analysis/result/qa_res/overall/2.2.8.1.json"
NEW = True

def get_f1(dic: dict):
    total_number = dic["tp"] + dic["fp"] + dic["fn"]
    precision = dic["precision_bev"]
    recall = dic["recall_bev"]
    f1 = 2 * precision * recall / (precision + recall)
    tp_number, yaw, pos = get_error(dic)
    
    return total_number, f1, tp_number, yaw, pos

def get_f1_old(dic: dict, res: dict):
    total_number = dic["tp"] + dic["fp"] + dic["fn"]
    yaw = dic["yaw_error"]["sigma2"]
    pos = dic["position_error"]["sigma2"]
    f1 = res["all_P0_bev"]["f1_score"]
    tp_number = dic["yaw_error"]["count"]
    
    return total_number, f1, tp_number, yaw, pos
    

def get_error(dic: dict):
    number = dic["yaw_error"]["count"]
    yaw = dic["yaw_error"]["sigma2"]
    pos = dic["position_error"]["sigma2"]
    return number, yaw, pos
    
with open(PATH, 'r') as f:
    json_obj = json.load(f)
    
    if NEW:
        selected_area = json_obj["-20m_20m"]
        vehicle, pedestrian, cyclist = selected_area.values()
        vehicle_no, vehicle_f, vehicle_tp, vehicle_yaw, vehicle_pos = get_f1(vehicle)
        pedestrian_no, pedestrian_f, pedestrian_tp, pedestrian_yaw, pedestrian_pos = get_f1(pedestrian)
        cyclist_no, cyclist_f, cyclist_tp, cyclist_yaw, cyclist_pos = get_f1(cyclist)
        
                
        weighted_f = (vehicle_f * vehicle_no + pedestrian_f * pedestrian_no + cyclist_f * cyclist_no) / (vehicle_no + cyclist_no + pedestrian_no)
        weighted_yaw = (vehicle_yaw * vehicle_tp + pedestrian_yaw * pedestrian_tp + cyclist_yaw * cyclist_tp) / (vehicle_tp + cyclist_tp + pedestrian_tp)
        weighted_pos = (vehicle_pos * vehicle_tp + pedestrian_pos * pedestrian_tp + cyclist_pos * cyclist_tp) / (vehicle_tp + cyclist_tp + pedestrian_tp)
    
    else:
        selected_area = json_obj["0-20m"]
        vehicle, vru, _ = selected_area.values()
        vehicle_no, vehicle_f, vehicle_tp, vehicle_yaw, vehicle_pos = get_f1_old(vehicle, json_obj)
        vru_no, vru_f, vru_tp, vru_yaw, vru_pos = get_f1_old(vru, json_obj)
        weighted_f = vehicle_f
        weighted_yaw = (vehicle_yaw * vehicle_tp + vru_yaw * vru_tp) / (vehicle_tp + vru_tp)
        weighted_pos = (vehicle_pos * vehicle_tp + vru_pos * vru_tp) / (vehicle_tp + vru_tp)

        

    print(PATH)
    print(weighted_f)
    print(weighted_yaw)
    print(weighted_pos)
