from pyexpat import model
from src.data_manager.data_manager_creator import data_manager_creator
import pandas as pd

class Config:
    ROOT = '/cpfs/output/large_vehicle_compare'
    EVAL_DATAFRAME_PATH_ORI = '%s/dataframes/eval_dataframe_ori.pkl' % ROOT
    EVAL_DATAFRAME_PATH_UP = '%s/dataframes/eval_dataframe_up.pkl' % ROOT
    
class EvalDataFrameConfigOri(Config):
    JSON_PATH = '/data_path/qa_M-PTN-139-CONTROL_beibao.txt' 
    JSON_TYPE = "txt"
    DATA_TYPE = "qa_cam3d"
    
class EvalDataFrameConfigUp(Config):
    JSON_PATH = '/data_path/qa_M-PT188-139_beibao.txt'
    JSON_TYPE = "txt"
    DATA_TYPE = "qa_cam3d"


def metric_calculator(df: pd.DataFrame, model: str) -> dict:
    res_dict = dict()
    res_dict["name"] = model
    metric_flag = df.flag.value_counts()
    tp = metric_flag["good"]
    fp = metric_flag["false"]
    fn = metric_flag["miss"]
    res_dict["precision"] = 100* tp / (tp + fp)
    res_dict["recall"] = 100* tp / (tp + fn)
    res_dict["f1"] = 2 * res_dict["precision"] * res_dict["recall"] / (res_dict["precision"] + res_dict["recall"])
    
    res_dict["yaw_diff"] = df.yaw_diff.mean()
    res_dict["dis_diff"] = df.euclidean_dis_diff.mean()
    res_dict["dis_diff_ratio"] = 100 * df.dis_ratio.mean()
    
    return res_dict
    

if __name__ == '__main__':
    # ori = data_manager_creator(EvalDataFrameConfigOri)
    # ori.load_from_json()
    # ori.save_to_pickle(EvalDataFrameConfigOri.EVAL_DATAFRAME_PATH_ORI)
            
    # up = data_manager_creator(EvalDataFrameConfigUp)
    # up.load_from_json()
    # up.save_to_pickle(EvalDataFrameConfigUp.EVAL_DATAFRAME_PATH_UP)

    combine_res = []
    
    ori_df = pd.read_pickle(EvalDataFrameConfigOri.EVAL_DATAFRAME_PATH_ORI)["df"]
    up_df = pd.read_pickle(EvalDataFrameConfigUp.EVAL_DATAFRAME_PATH_UP)["df"]
    
    ori_df = ori_df[(ori_df.class_name.isin(["bus", "truck"]) & (abs(ori_df.x)) < 8) & (1 < abs(ori_df.y)) & (abs(ori_df.y) < 5)]
    up_df = up_df[(up_df.class_name.isin(["bus", "truck"]) & (abs(up_df.x)) < 8) & (1 < abs(up_df.y)) & (abs(up_df.y) < 5)]
    
    combine_res.append(metric_calculator(ori_df, "ori"))
    combine_res.append(metric_calculator(up_df, "up"))
    
    df = pd.DataFrame(combine_res)
    df = df.set_index("name")
    df = df.sort_values("name")
    
    df = df.T
    df["diff"] = df["up"] - df["ori"]
    # df["diff_ratio"] = 100 * abs(df["diff"]) / df["ori"]
    df = df.round(4).T
    print(df)
    

