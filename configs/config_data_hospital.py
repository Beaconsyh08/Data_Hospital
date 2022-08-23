NAME = "v31_total"
class Config:
    # ROOT = '/share/analysis/result/data_hospital_data/0628/%s' % NAME
    ROOT = '/root/data_hospital_data/0728v60/%s' % NAME
    LOGISTIC_DATAFRAME_PATH = '%s/dataframes/logistic_dataframe.pkl' % ROOT
    REPROJECT_DATAFRAME_PATH = '%s/dataframes/reproject_dataframe.pkl' % ROOT
    FINAL_DATAFRAME_PATH = '%s/dataframes/final_dataframe.pkl' % ROOT
    BADCASE_DATAFRAME_PATH = '%s/dataframes/badcase_dataframe.pkl' % ROOT
    
    TYPE_MAP = {'car': 'car', 'van': 'car', 
                'truck': 'truck', 'forklift': 'truck',
                'bus':'bus', 
                'rider':'rider',
                'rider-bicycle': 'rider', 'rider-motorcycle':'rider', 
                'bicycle': 'bicycle', 'motorcycle': 'bicycle',
                'tricycle': 'tricycle', 'closed-tricycle':'tricycle', 'open-tricycle': 'tricycle', 'closed_tricycle':'tricycle', 'open_tricycle': 'tricycle', 'pedestrian': 'pedestrian',
                'static': 'static', 'trafficCone': 'static', 'water-filledBarrier': 'static', 'other': 'static', 'accident': 'static', 'construction': 'static', 'traffic-cone': 'static', 'other-vehicle': 'static', 'attached': 'static', 'accident': 'static', 'traffic_cone': 'static', 'other-static': 'static', 'water-filled-barrier': 'static', 'other_static': 'static', 'water_filled_barrier': 'static', 'dynamic': 'static', 'other_vehicle': 'static', 'trafficcone': 'static', 'water-filledbarrier': 'static',
                }


class DataHospitalConfig(Config):
    # MODULES = ["Duplicated", "Logistic", "Reproject", "CoorTrans", "Inference"]
    # MODULES = ["Logistic", "Reproject", "CoorTrans", "Inference"]
    # MODULES = ["Inference"]
    MODULES = []
    ORIENTATION = "SIDE"
    COOR = "Lidar"
    VIS = False
    
    
class DataHospital2Config(Config):
    MODULES = ["Evaluate"]


class DuplicatedDoctorConfig(Config):
    JSON_PATH = "/data_path/%s.txt" % NAME
    ORI_JSON_PATH = "/data_path/%s.txt" % NAME
    ORI_PKL_PATH = "/root/data_hospital_data/dataframes/sidecam_ori.pkl"
    SAVE_PKL_PATH = "/root/data_hospital_data/dataframes/%s.pkl" % NAME
    PKL_READY = False
    METHOD = "Total"
    
    
class LogisticDoctorConfig(Config):
    JSON_PATH = '%s/duplicated_doctor/clean.txt' % Config.ROOT
    JSON_TYPE = "txt"
    DATA_TYPE = "train_cam3d"
    ERROR_LIST = ["bbox_error", "coor_error"]
    SAVE_DIR = "%s/logistic_doctor" % Config.ROOT
    COOR = DataHospitalConfig.COOR
    ONLINE = False
    VIS = True
    
    
class ReprojectDoctorConfig(Config):
    JSON_PATH = '%s/logistic_doctor/clean.txt' % Config.ROOT
    JSON_TYPE = "txt"
    DATA_TYPE = "train_cam3d"
    SAVE_DIR = "%s/reproject_doctor" % Config.ROOT
    LOAD_PATH = "%s/ready_2_reproject" % SAVE_DIR
    VIS_PATH = "%s/vis" % SAVE_DIR
    THRESHOLD = 0.1
    COOR = DataHospitalConfig.COOR
    VIS = DataHospitalConfig.VIS
    
    
class CoorTransConfig(Config):
    OUTPUT_DIR = '%s/coor_trans_doctor/trans/' % Config.ROOT
    INPUT_PATH = '%s/reproject_doctor/clean.txt' % Config.ROOT
    OUTPUT_PATH = '%s/coor_trans_doctor/to_be_inf.txt' % Config.ROOT
    MAPPING = True
    

class InferenceConfig(Config):
    INF_OUTPUT_DIR = '%s/inference_doctor/'% Config.ROOT
    INF_MODEL_PATH = '/share/analysis/syh/models/2.2.8.0-0811-M-PT288-221-U.pth'
    
    
class EvaluateConfig(Config):
    NAME = NAME
    MODEL_NAME = InferenceConfig.INF_MODEL_PATH[:-4]
    INPUT_DIR = '%s/inference_doctor/'% Config.ROOT
    OUTPUT_DIR = '%s/evaluate_doctor' % Config.ROOT
    
    JSON_PATH = '/data_path/data_hospital_badcase.txt'
    JSON_TYPE = "txt"
    DATA_TYPE = "qa_cam3d_temp"
    
    
class MissAnnoDoctorConfig(Config):
    SAVE_DIR = "%s/miss_anno_doctor" % Config.ROOT
    
    
class MatchingDoctorConfig(Config):
    SAVE_DIR = "%s/matching_doctor" % Config.ROOT


class StatsDoctorConfig(Config):
    SAVE_DIR = "%s/stats_doctor" % Config.ROOT
    
    
class LogConfig(Config):
    LOG_PATH = '/share/analysis/log/analysis.log'


class OutputConfig(Config):
    OUTPUT_DIR = '/cpfs/output'
    OUTPUT_ANNOTATION_PATH = '%s/card/annotation' % OUTPUT_DIR


class VisualizationConfig(Config):
    SAVE_DIR = "%s/logistic_doctor/images/" % Config.ROOT
    