from configs.config import DataHospitalConfig, DuplicatedDoctorConfig, StatsDoctorConfig, LogicalCheckerConfig, ReprojectDoctorConfig, CoorTransConfig, InferenceConfig
from src.data_hospital.coor_trans_doctor import CoorTransDoctor
from src.data_hospital.reproject_doctor import ReprojectDoctor
from src.data_hospital.logical_checker import LogicalChecker
from src.data_hospital.duplicated_doctor import DuplicatedDoctor
from src.data_hospital.stats_doctor import StatsDoctor
from src.utils.logger import get_logger
import os

logger = get_logger()

class DataHospital():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.modules = cfg.MODULES
        
    def run(self, ) -> None:
        
        logger.critical("Data Hospital I Running")
        logger.error("Activated Modules: %s" % ', '.join(map(str, self.modules)))
        
        if "Duplicated" in self.modules:
            logger.critical("Duplicated Doctor Diagnosing")
            duplicated_doctor = DuplicatedDoctor(DuplicatedDoctorConfig)
            duplicated_doctor.self_diagnose()
        
        if "Logical" in self.modules:
            logger.critical("Logical Doctor Diagnosing")
            logical_doctor = LogicalChecker(LogicalCheckerConfig)
            logical_doctor.diagnose()
            logical_doctor.txt_for_reproejct()
            
        if "Reproject" in self.modules:
            logger.critical("Reproject Doctor Diagnosing")
            reproject_doctor = ReprojectDoctor(ReprojectDoctorConfig)
            reproject_doctor.diagnose()          
            
        if "CoorTrans" in self.modules:
            logger.critical("Coor Trans Doctor Diagnosing")
            coor_trans_doctor = CoorTransDoctor(CoorTransConfig)
            coor_trans_doctor.diagnose()
        
        logger.critical("Data Hospital I Completed")
        

if __name__ == '__main__':
    data_hospital = DataHospital(DataHospitalConfig)
    data_hospital.run()
    

    
    # logger.critical("Stats Doctor Diagnosing")
    # stats_doctor = StatsDoctor(StatsDoctorConfig)
    # stats_doctor.diagnose()
    
    
