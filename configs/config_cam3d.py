class Config:
    NAME1 = '0803'
    # ROOT = '/share/analysis/cmh/result'
    ROOT = '/cpfs/output/%s/other' % NAME1
    EMB_PATH = '%s/embedding/feas_moco_512_l2.npy' % ROOT
    DATAFRAME_PATH = '%s/dataframes/combine_dataframe.pkl' % ROOT
    DATAFRAME_PATH_P0 = '%s/dataframes/P0_dataframe.pkl' % ROOT
    EVAL_DATAFRAME_PATH = '%s/dataframes/eval_dataframe.pkl' % ROOT
    TRAIN_DATAFRAME_PATH = '%s/dataframes/train_dataframe.pkl' % ROOT
    RETRIEVAL_DATAFRAME_PATH = '%s/dataframes/retrieval_dataframe.pkl' % ROOT
    INFERENCE_DATAFRAME_PATH = '%s/dataframes/inference_dataframe.pkl' % ROOT


class OutputConfig(Config):
    OUTPUT_DIR = '/cpfs/output/%s' % Config.NAME1
    QUERY_OUTPUT_DIR = '%s/card/query' % OUTPUT_DIR
    OTHERS_OUTPUT_DIR = '%s/other' % OUTPUT_DIR
    DATAFRAME_OUTPUT_PATH = '%s/other/analysis_result.pkl' % OUTPUT_DIR
    OUTPUT_ANNOTATION_PATH = '%s/card/annotation' % OUTPUT_DIR


class EvalDataFrameConfig(Config):
    JSON_PATH = '/data_path/225_badcase.txt'
    JSON_TYPE = "txt"
    DATA_TYPE = "qa_cam3d"


class TrainDataFrameConfig(Config):
    JSON_PATH = "/data_path/train9999999.txt"
    JSON_TYPE = "txt"
    DATA_TYPE = "train_cam3d"

class LogicalCheckerConfig(Config):
    ERROR_LIST = ["bbox_error", "coor_error"]
    # ERROR_LIST = ["quality_error"]
    IGNORE_LIST = []
    MF = "Miss" # ["Miss", "False"]
    THRESHOLD = 0.3
    COOR = "Vehicle"
    
    
# class SamplerConfig(Config):
#     SAMPLE_TYPE = 'uniform'    # uniform, kmpp
#     NUM_SAMPLES = 1000000
#     SAMPLE_COLUMN = 'flag'
#     SAMPLE_DATA_TYPE = 'false'
#     CLUSTER_NAME = 'cluster_id'


class SamplerConfig(Config):
    SAMPLE_TYPE = 'uniform'    # least_confidence
    NUM_SAMPLES = 5000
    THRESHOLD = 0.7
    SAMPLE_COLUMN = 'flag'
    SAMPLE_DATA_TYPE = 'good'
    CLUSTER_NAME = 'cluster_id'
    TYPE = "threshold"      # choose from ["threshold ", "amount"]
    
    
class ClusterConfig(Config):
    CLUSTER_METHOD = "dbscan"    # kmeans, dbscan...
    PCA_VAR_RATIO = 0.95


class EmbeddingConfig(Config):
    CONFIG_PATH = "./src/feature_embedding/config/embedding_3_channel.py"
    SAVE_PATH = "%s/embedding" % Config.ROOT


class LogConfig(Config):
    LOG_PATH = '/share/analysis/log/analysis.log'


class StatsConfig(Config):
    STATS_TYPE = 'cam3d'
    ALL_CATEGORY = ['car', 'truck', 'bus', 'pedestrian', 'rider', 'tricycle', 'bicycle']
    # ALL_CATEGORY = ['car', 'bus', 'truck', 'van', 'forklift', 'pedestrian', 'rider', 'bicycle', 'tricycle', 'rider-bicycle', 'rider-motorcycle', 'rider_bicycle', 'rider_motorcycle', 'closed-tricycle', 'open-tricycle', 'closed_tricycle', 'open_tricycle']

    STATS_DATAFRAME_PATH = "%s/dataframes/eval_dataframe.pkl" % Config.ROOT
    MULTI_VERSIONS = [ '/data_path/226_badcase.txt',
                        '/data_path/225_badcase.txt',
                    #   '/data_path/qa_188-139.txt'
                    #    '/data_path/eval_badcase.txt' 
                    ]
    SIGMA1 = 0.6526
    SIGMA2 = 0.9544
    SIGMA3 = 0.9974
    

class VisualizationConfig(Config):
    SAVE_DIR = '/cpfs/output/%s/other' % Config.NAME1
    MAX_TSNE_SAMPLES = 20000


# class RetrievalConfig(Config):
#     RETRIEVAL_RESULT_PATH = '/data_path/retrieval.txt'
#     TOP_K = 30      # Each query has a maximum of 30 results, 0 < top_k <=30 
#     NUM_ANNOTATION = 100000


class RetrievalDataFrameConfig(Config):
    JSON_PATH = '/data_path/retrieval.txt'
    JSON_TYPE = "txt"
    DATA_TYPE = "ret_cam3d"
    

class RetrievalConfig(Config):
    RESULT_PATH = '/data_path/inference.txt'


class InferenceDataFrameConfig(Config):
    JSON_PATH = '/data_path/inference.txt'
    JSON_TYPE = "txt"
    DATA_TYPE = "inf_cam2d_cloud"


class AnnotationConfig(Config):
    CLASS_WEIGHT = {'pedestrian': 1, 'rider': 1, 'bicycle': 0.2, 'car': 0, 'bus': 0.5, 'truck': 1, 'tricycle': 1}
    SAMPLE_TYPE = "amount"
    SAMPLE_NUM = 100000
    THRESHOLD = 0.0
    DATA_TYPE = "inf_cam2d_cloud"
    DIMENSION = "group"
    ANNO_TXT_PATH = "/root/2d_analysis/src/exp/dataset/active_0.txt"

class DvcConfig(Config):
    DVC_PATH="./dvc/cam3d"

