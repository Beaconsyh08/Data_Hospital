NAME = "front_night"

import os

class Config:
    # ROOT = '/share/analysis/result/data_hospital_data/0628/%s' % NAME
    ROOT = '/root/data_hospital_data/%s' % NAME
    DATAFRAME_DIR = "%s/dataframes" % (ROOT)
    os.makedirs(DATAFRAME_DIR, exist_ok=True)
    DATAFRAME_PATH = '%s/dataframes/dataframe.pkl' % ROOT
    BADCASE_DATAFRAME_PATH = '%s/dataframes/badcase_dataframe.pkl' % ROOT
    FINAL_DATAFRAME_PATH = '%s/dataframes/final_dataframe.pkl' % ROOT
    TEMP_DIR = '%s/temp_dataframes' % ROOT 
    JSON_PATH = "/data_path/%s.txt" % NAME
    
    TYPE_MAP = {
            'car': 'car', 'van': 'car', 
            'truck': 'truck', 'forklift': 'truck',
            'bus':'bus', 
            'rider':'rider', 'rider_bicycle': 'rider', 'rider_motorcycle': 'rider',
            'rider-bicycle': 'rider', 'rider-motorcycle':'rider', 
            'bicycle': 'bicycle', 'motorcycle': 'bicycle',
            'tricycle': 'tricycle', 'closed-tricycle':'tricycle', 'open-tricycle': 'tricycle', 'closed_tricycle':'tricycle', 'open_tricycle': 'tricycle', 'pedestrian': 'pedestrian',
            'static': 'static', 'trafficCone': 'static', 'water-filledBarrier': 'static', 'other': 'static', 'accident': 'static', 'construction': 'static', 'traffic-cone': 'static', 
            'other-vehicle': 'static', 'attached': 'static', 'traffic_cone': 'static', 'other-static': 'static', 'water-filled-barrier': 'static', 'other_static': 'static', 'water_filled_barrier': 'static', 
            'dynamic': 'static', 'other_vehicle': 'static', 'trafficcone': 'static', 'water-filledbarrier': 'static',
            }
    CATEGORY = {
            'vehicle': ['car', 'bus', 'truck', 'van', 'forklift'],
            'vru': ['pedestrian', 'rider', 'bicycle', 'tricycle', 'rider-bicycle', 'motorcycle', 'rider-motorcycle', 'rider_bicycle', 'rider_motorcycle', 'closed-tricycle', 'open-tricycle', 'closed_tricycle', 'open_tricycle'],
            'static': ['static', 'trafficCone', 'water-filledBarrier', 'other', 'accident', 'construction', 'traffic-cone', 'other-vehicle', 'attached', 'traffic_cone', 'other-static', 'water-filled-barrier', 'other_static', 
                        'water_filled_barrier', 'dynamic', 'other_vehicle', 'water-filledbarrier']
            }



class DataHospitalConfig(Config):
    # Full Data Checking 
    MODULES = ["Logical", "Calibration", "CoordinateConverter", "Inference", "Evaluate", "MissAnno", "Matching", "Statistics"]
    
    # Data Checking Phase I
    # MODULES = ["Duplicated", "Logical", "Calibration", "Statistics"]
    
    
    # Data Checking Phase II
    # MODULES = ["CoordinateConverter", "Inference", "Evaluate", "MissAnno", "Matching", "Statistics"]
    
    # Fast Single Frame
    # MODULES = ["Duplicated", "Logical", "Statistics"]
    
    # Inf & Eval
    # MODULES = ["Logical", "Calibration", "Statistics"]
    # MODULES = ["Logical", "Statistics"]
    # MODULES = ["CoordinateConverter", "Inference", "Evaluate"]
    EVALUATOR = "LUCAS"
    COOR = "Vehicle"
    
    # MODULES = ["Statistics"]
    
    TOTAL_ERROR_LIST = ["dup_json", "dup_img", "empty", "bbox_error", "coor_error", "res_error", "outlier_error", "calibration_error", "miss_anno_error", "matching_error"]
    ORIENTATION = "SIDE"
    VIS = False
    
    
class DuplicatedCheckerConfig(Config):
    JSON_PATH = Config.JSON_PATH
    ORI_JSON_PATH = Config.JSON_PATH
    SAVE_DIR = "%s/duplicated_checker" % Config.ROOT
    ORI_PKL_PATH = "%s/dataframes/sidecam_ori.pkl" % Config.ROOT
    SAVE_PKL_PATH = "%s/dataframes/%s.pkl" % (Config.ROOT, NAME)
    PKL_READY = False
    METHOD = "Total"
    
    
class LogicalCheckerConfig(Config):
    JSON_PATH = Config.JSON_PATH
    JSON_TYPE = "txt"
    DATA_TYPE = "train_cam3d"
    ERROR_LIST = ["bbox_error", "coor_error", "res_error"]
    SAVE_DIR = "%s/logical_checker" % Config.ROOT
    COOR = DataHospitalConfig.COOR
    ONLINE = False
    VIS = False


class OutlierCheckerConfig(Config):
    JSON_PATH = "../data_hospital_data/%s/logical_checker/clean.txt" % (NAME)
    JSON_PATH = Config.JSON_PATH if not os.path.exists(JSON_PATH) else JSON_PATH
    JSON_TYPE = "txt"
    DATA_TYPE = "train_cam3d"
    SAVE_DIR = "%s/outlier_checker" % Config.ROOT
    VIS = False
    
    
class CalibrationCheckerConfig(Config):
    JSON_PATH = Config.JSON_PATH
    JSON_TYPE = "txt"
    DATA_TYPE = "train_cam3d"
    SAVE_DIR = "%s/calibration_checker" % Config.ROOT
    LOAD_PATH = "%s/ready_2_calibration" % SAVE_DIR
    VIS_PATH = "%s/vis" % SAVE_DIR
    THRESHOLD = 0.1
    COOR = DataHospitalConfig.COOR
    VIS = DataHospitalConfig.VIS
    
    
class CoordinateConverterConfig(Config):
    OUTPUT_DIR = '%s/coordinate_converter/trans/' % Config.ROOT
    INPUT_PATH = Config.JSON_PATH
    OUTPUT_PATH = '%s/coordinate_converter/to_be_inf.txt' % Config.ROOT
    MAPPING = True
    

class InferenceConfig(Config):
    INF_OUTPUT_DIR = '%s/Data_Inferencer/'% Config.ROOT
    # INF_MODEL_PATH = '/share/analysis/syh/models/2.2.0.0-0616-M-PT188-192-U.pth'
    # INF_MODEL_PATH = '/share/analysis/syh/models/2.2.8.0-0811-M-PT288-221-U.pth'
    INF_MODEL_PATH = '/share/analysis/syh/models/BASE20+FN2.pth'
    
    # INF_MODEL_PATH = '/share/analysis/syh/models/clean50.pth'
    
    
class EvaluateProcessorConfig(Config):
    NAME = NAME
    MODEL_NAME = InferenceConfig.INF_MODEL_PATH[:-4]
    INPUT_DIR = '%s/Data_Inferencer'% Config.ROOT
    OUTPUT_DIR = '%s/evaluate_processor' % Config.ROOT
    JSON_PATH = '%s/data_hospital_badcase.txt' % OUTPUT_DIR
    JSON_TYPE = "txt"
    DATA_TYPE = "qa_cam3d" if DataHospitalConfig.EVALUATOR == "LUCAS" else "qa_cam3d_temp"
    
    
class MissAnnoCheckerConfig(Config):
    SAVE_DIR = "%s/miss_anno_checker" % Config.ROOT
    VIS = DataHospitalConfig.VIS
    
    
class MatchingCheckerConfig(Config):
    SAVE_DIR = "%s/matching_checker" % Config.ROOT
    VIS = DataHospitalConfig.VIS


class OutputConfig(Config):
    OUTPUT_DIR = '/cpfs/output'
    OUTPUT_CARD_DIR = '%s/card' % OUTPUT_DIR


class StatisticsManagerConfig(Config):
    JSON_PATH = Config.JSON_PATH
    JSON_TYPE = "txt"
    DATA_TYPE = "train_cam3d"
    SAVE_DIR = "%s/statistics_manager" % OutputConfig.OUTPUT_CARD_DIR
    
    
class LogConfig(Config):
    LOG_PATH = '/share/analysis/log/analysis.log'


class VisualizationConfig(Config):
    SAVE_DIR = "%s/logical_checker/images/" % Config.ROOT
    