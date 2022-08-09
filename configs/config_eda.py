class Config:
    NAME = "lucas_beibao"
    ROOT = '/share/analysis/result/eda'
    DATAFRAME_PATH = '%s/dataframes/%s.pkl' % (ROOT, NAME)
    # DATAFRAME_PATH = "/share/analysis/result/lucas_data_cleaner/dataframes/train_dataframe.pkl"
    DF_READY = False
    
    
class EDAConfig(Config):
    SAVE_DIR = "%s/%s" % (Config.ROOT, Config.NAME)
    XY_PLOT_DIM = ["camera_orientation", "priority", "flag"]

class DataFrameConfig(Config):
    DATA_TYPE = "train_cam3d"
    JSON_TYPE = "txt"
    # JSON_PATH = '/root/2d_analysis/src/exp/dataset/%s.txt' % Config.NAME
    JSON_PATH = '/data_path/%s.txt' % Config.NAME

class LogConfig(Config):
    LOG_PATH = '/share/analysis/log/analysis.log'
    
class EmbeddingConfig(Config):
    CONFIG_PATH = "./src/feature_embedding/config/embedding_3_channel.py"
    SAVE_PATH = "%s/embedding" % Config.ROOT