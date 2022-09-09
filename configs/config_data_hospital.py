NAME = "v71_test"
class Config:
    # ROOT = '/share/analysis/result/data_hospital_data/0628/%s' % NAME
    ROOT = '/root/data_hospital_data/%s' % NAME
    LOGICAL_DATAFRAME_PATH = '%s/dataframes/logical_dataframe.pkl' % ROOT
    CALIBRATION_DATAFRAME_PATH = '%s/dataframes/calibration_dataframe.pkl' % ROOT
    FINAL_DATAFRAME_PATH = '%s/dataframes/final_dataframe.pkl' % ROOT
    BADCASE_DATAFRAME_PATH = '%s/dataframes/badcase_dataframe.pkl' % ROOT
    BADCASE_DATAFRAME_PATH_DMISS = '%s/dataframes/badcase_dmiss_dataframe.pkl' % ROOT
    BADCASE_DATAFRAME_PATH_DMATCH = '%s/dataframes/badcase_dmiss_dataframe.pkl' % ROOT
    VIS_DATAFRAME_PATH = '%s/dataframes/vis.pkl' % ROOT
    
    TYPE_MAP = {'car': 'car', 'van': 'car', 
                'truck': 'truck', 'forklift': 'truck',
                'bus':'bus', 
                'rider':'rider', 'rider_bicycle': 'rider', 'rider_motorcycle': 'rider',
                'rider-bicycle': 'rider', 'rider-motorcycle':'rider', 
                'bicycle': 'bicycle', 'motorcycle': 'bicycle',
                'tricycle': 'tricycle', 'closed-tricycle':'tricycle', 'open-tricycle': 'tricycle', 'closed_tricycle':'tricycle', 'open_tricycle': 'tricycle', 'pedestrian': 'pedestrian',
                'static': 'static', 'trafficCone': 'static', 'water-filledBarrier': 'static', 'other': 'static', 'accident': 'static', 'construction': 'static', 'traffic-cone': 'static', 'other-vehicle': 'static', 'attached': 'static', 'accident': 'static', 'traffic_cone': 'static', 'other-static': 'static', 'water-filled-barrier': 'static', 'other_static': 'static', 'water_filled_barrier': 'static', 'dynamic': 'static', 'other_vehicle': 'static', 'trafficcone': 'static', 'water-filledbarrier': 'static',
                }


class DataHospitalConfig(Config):
    MODULES = ["Duplicated", "Logical", "Calibration", "CoordinateConverter", "Inference", "Evaluate", "MissAnno", "Matching", "Statistics"]
    # MODULES = ["Duplicated"]
    EVALUATOR = "LUCAS"
    ORIENTATION = "SIDE"
    COOR = "Lidar"
    VIS = False
    
    
class DuplicatedCheckerConfig(Config):
    JSON_PATH = "/data_path/%s.txt" % NAME
    ORI_JSON_PATH = "/data_path/%s.txt" % NAME
    ORI_PKL_PATH = "/root/data_hospital_data/dataframes/sidecam_ori.pkl"
    SAVE_PKL_PATH = "/root/data_hospital_data/dataframes/%s.pkl" % NAME
    PKL_READY = False
    METHOD = "Total"
    
    
class LogicalCheckerConfig(Config):
    JSON_PATH = '%s/duplicated_checker/clean.txt' % Config.ROOT
    JSON_TYPE = "txt"
    DATA_TYPE = "train_cam3d"
    ERROR_LIST = ["bbox_error", "coor_error", "res_error"]
    CHECK_ERROR_LIST = ERROR_LIST + ["2d_null_error", "3d_null_error"]
    SAVE_DIR = "%s/logical_checker" % Config.ROOT
    COOR = DataHospitalConfig.COOR
    ONLINE = False
    VIS = False
    
    
class CalibrationCheckerConfig(Config):
    JSON_PATH = '%s/logical_checker/clean_empty.txt' % Config.ROOT
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
    INPUT_PATH = '%s/calibration_checker/clean.txt' % Config.ROOT
    OUTPUT_PATH = '%s/coordinate_converter/to_be_inf.txt' % Config.ROOT
    MAPPING = True
    

class InferenceConfig(Config):
    INF_OUTPUT_DIR = '%s/data_inferencer/'% Config.ROOT
    # INF_MODEL_PATH = '/share/analysis/syh/models/2.2.8.0-0811-M-PT288-221-U.pth'
    INF_MODEL_PATH = '/share/analysis/syh/models/clean50.pth'
    
    
class EvaluateProcessorConfig(Config):
    NAME = NAME
    MODEL_NAME = InferenceConfig.INF_MODEL_PATH[:-4]
    INPUT_DIR = '%s/data_inferencer/'% Config.ROOT
    OUTPUT_DIR = '%s/evaluate_processor' % Config.ROOT
    
    JSON_PATH = '/data_path/data_hospital_badcase.txt'
    JSON_TYPE = "txt"
    DATA_TYPE = "qa_cam3d" if DataHospitalConfig.EVALUATOR == "LUCAS" else "qa_cam3d_temp"
    
    
class MissAnnoCheckerConfig(Config):
    SAVE_DIR = "%s/miss_anno_checker" % Config.ROOT
    VIS = DataHospitalConfig.VIS
    
    
class MatchingCheckerConfig(Config):
    SAVE_DIR = "%s/matching_checker" % Config.ROOT
    VIS = DataHospitalConfig.VIS


class StatisticsManagerConfig(Config):
    JSON_PATH = '%s/matching_checker/clean.txt' % Config.ROOT
    JSON_TYPE = "txt"
    DATA_TYPE = "train_cam3d"
    SAVE_DIR = "%s/statistics_manager" % Config.ROOT
    
    
class LogConfig(Config):
    LOG_PATH = '/share/analysis/log/analysis.log'


class OutputConfig(Config):
    OUTPUT_DIR = '/cpfs/output'
    OUTPUT_ANNOTATION_PATH = '%s/card/annotation' % OUTPUT_DIR


class VisualizationConfig(Config):
    SAVE_DIR = "%s/logical_checker/images/" % Config.ROOT
    