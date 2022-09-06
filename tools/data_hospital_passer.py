from configs.config import DataHospitalConfig, InferenceConfig, CoorTransConfig
from src.utils.logger import get_logger


logger = get_logger()

class DataHospitalPasser():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.modules = cfg.MODULES
        
    def run(self, ) -> None:
        
        logger.critical("Data Hospital Passer Running")
        
        if "Inference" in self.modules and self.cfg.ORIENTATION == "SIDE":
            print(DataHospitalConfig.ORIENTATION)
            print(InferenceConfig.INF_MODEL_PATH)
            print(CoorTransConfig.OUTPUT_PATH)
            print(InferenceConfig.INF_OUTPUT_DIR)
            print("inference")
        
if __name__ == '__main__':
    data_hospital_passer = DataHospitalPasser(DataHospitalConfig)
    data_hospital_passer.run()
    

    
