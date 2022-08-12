from configs.config import DataHospital2Config, CoorTransConfig
from src.data_hospital.coor_trans_doctor import CoorTransDoctor
from src.utils.logger import get_logger
import os


logger = get_logger()

class DataHospital2():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.modules = cfg.MODULES
        
    def run(self, ) -> None:
        
        logger.critical("Data Hospital II Running")
        
        if "CoorTrans" in self.modules:
            logger.critical("Coor Trans Doctor Diagnosing")
            coor_trans_doctor = CoorTransDoctor(CoorTransConfig)
            coor_trans_doctor.diagnose()
        
        if "Inference" in self.modules:
            logger.critical("Inferencing...")
            os.system("../data_inferencer/run.sh")
        
        
if __name__ == '__main__':
    data_hospital2 = DataHospital2(DataHospital2Config)
    data_hospital2.run()
    
