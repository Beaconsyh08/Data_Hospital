
from configs.config import (Config, EDAConfig, DataFrameConfig)
from src.exp.exploratory_data_analysis import ExploratoryDataAnalysis
from src.data_manager.data_manager_creator import data_manager_creator


def main():
    if not EDAConfig.DF_READY:
        data_manager = data_manager_creator(DataFrameConfig)
        data_manager.load_from_json()
        data_manager.save_to_pickle(Config.DATAFRAME_PATH)
    
    eda = ExploratoryDataAnalysis(EDAConfig)
    eda.process()

if __name__ == '__main__':
    main()