NAME = "qa_cmh_1000"
class Config:
    # ROOT = '/share/analysis/result/data_hospital_data/0628/%s' % NAME
    ROOT = '/root/data_hospital_data/0728v60/%s' % NAME
    LOGISTIC_DATAFRAME_PATH = '%s/dataframes/logistic_dataframe.pkl' % ROOT
    REPROJECT_DATAFRAME_PATH = '%s/dataframes/reproject_dataframe.pkl' % ROOT
    FINAL_DATAFRAME_PATH = '%s/dataframes/final_dataframe.pkl' % ROOT


class DataHospitalConfig(Config):
    MODULES = ["Duplicated", "Logistic", "Reproject"]


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
    COOR = "Car"
    ONLINE = False
    VIS = False
    
    
class ReprojectDoctorConfig(Config):
    JSON_PATH = '%s/logistic_doctor/clean.txt' % Config.ROOT
    JSON_TYPE = "txt"
    DATA_TYPE = "train_cam3d"
    SAVE_DIR = "%s/reproject_doctor" % Config.ROOT
    LOAD_PATH = "%s/ready_2_reproject" % SAVE_DIR
    VIS_PATH = "%s/vis" % SAVE_DIR
    THRESHOLD = 0.1
    COOR = "Car"
    VIS = False
    

class StatsDoctorConfig(Config):
    SAVE_DIR = "%s/stats_doctor" % Config.ROOT
    
    
class LogConfig(Config):
    LOG_PATH = '/share/analysis/log/analysis.log'


class OutputConfig(Config):
    OUTPUT_DIR = '/cpfs/output'
    OUTPUT_ANNOTATION_PATH = '%s/card/annotation' % OUTPUT_DIR


class VisualizationConfig(Config):
    SAVE_DIR = "%s/logistic_doctor/images/" % Config.ROOT
    