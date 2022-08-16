NAME = "qa_cmh"
class Config:
    # ROOT = '/share/analysis/result/data_hospital_data/0628/%s' % NAME
    ROOT = '/root/data_hospital_data/0728v60/%s' % NAME
    LOGISTIC_DATAFRAME_PATH = '%s/dataframes/logistic_dataframe.pkl' % ROOT
    REPROJECT_DATAFRAME_PATH = '%s/dataframes/reproject_dataframe.pkl' % ROOT
    FINAL_DATAFRAME_PATH = '%s/dataframes/final_dataframe.pkl' % ROOT


class DataHospitalConfig(Config):
    # MODULES = ["Duplicated", "Logistic", "Reproject", "CoorTrans", "Inference"]
    MODULES = [""]

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
    COOR = "Lidar"
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
    COOR = "Lidar"
    VIS = False
    
class CoorTransConfig(Config):
    OUTPUT_DIR = '%s/coor_trans_doctor/trans/' % Config.ROOT
    INPUT_PATH = '%s/reproject_doctor/clean.txt' % Config.ROOT
    OUTPUT_PATH = '%s/coor_trans_doctor/to_be_inf.txt' % Config.ROOT
    INF_OUTPUT_DIR = '%s/inference_doctor/'% Config.ROOT
    
class EvaluateConfig(Config):
    NAME = NAME
    INPUT_DIR = '%s/inference_doctor/'% Config.ROOT
    OUTPUT_DIR = '%s/evaluate_doctor' % Config.ROOT
    
class StatsDoctorConfig(Config):
    SAVE_DIR = "%s/stats_doctor" % Config.ROOT
    
    
class LogConfig(Config):
    LOG_PATH = '/share/analysis/log/analysis.log'


class OutputConfig(Config):
    OUTPUT_DIR = '/cpfs/output'
    OUTPUT_ANNOTATION_PATH = '%s/card/annotation' % OUTPUT_DIR


class VisualizationConfig(Config):
    SAVE_DIR = "%s/logistic_doctor/images/" % Config.ROOT
    