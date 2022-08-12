from configs.config import DataHospitalConfig, DuplicatedDoctorConfig, StatsDoctorConfig, LogisticDoctorConfig, ReprojectDoctorConfig
from src.data_hospital.reproject_doctor import ReprojectDoctor
from src.data_hospital.logistic_doctor import LogisticDoctor
from src.data_hospital.duplicated_doctor import DuplicatedDoctor
from src.data_hospital.stats_doctor import StatsDoctor
from src.utils.logger import get_logger

logger = get_logger()

class DataHospital():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.modules = cfg.MODULES
        
    def run(self, ) -> None:
        
        logger.critical("Data Hospital Running")
        
        if "Duplicated" in self.modules:
            logger.critical("Duplicated Doctor Diagnosing")
            duplicated_doctor = DuplicatedDoctor(DuplicatedDoctorConfig)
            duplicated_doctor.self_diagnose()
        
        if "Logistic" in self.modules:
            logger.critical("Logistic Doctor Diagnosing")
            logistic_doctor = LogisticDoctor(LogisticDoctorConfig)
            logistic_doctor.diagnose()
            logistic_doctor.txt_for_reproejct()
            
        if "Reproject" in self.modules:
            logger.critical("Reproject Doctor Diagnosing")
            reproject_doctor = ReprojectDoctor(ReprojectDoctorConfig)
            reproject_doctor.diagnose()          
            

if __name__ == '__main__':
    data_hospital = DataHospital(DataHospitalConfig)
    data_hospital.run()
    

    
    # logger.critical("Stats Doctor Diagnosing")
    # stats_doctor = StatsDoctor(StatsDoctorConfig)
    # stats_doctor.diagnose()
    
    
