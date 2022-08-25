from configs.config import DataHospitalConfig, EvaluateConfig, MissAnnoDoctorConfig
from configs.config_data_hospital import MatchingDoctorConfig
from src.data_hospital.evaluate_doctor import EvaluateDoctor
from src.data_hospital.matching_doctor import MatchingDoctor
from src.data_hospital.miss_anno_doctor import MissAnnoDoctor
from src.utils.logger import get_logger


logger = get_logger()

class DataHospital2():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.modules = cfg.MODULES
        
    def run(self, ) -> None:
        
        logger.critical("Data Hospital II Running")
        logger.error("Activated Modules: %s" % ', '.join(map(str, self.modules)))
        
        if "Inference" in self.modules:
            evaluate_doctor = EvaluateDoctor(EvaluateConfig)
            evaluate_doctor.diagnose()
        
        if "MissAnno" in self.modules:
            miss_anno_doctor = MissAnnoDoctor(MissAnnoDoctorConfig)
            miss_anno_doctor.diagnose()
            
        if "Matching" in self.modules:
            matching_doctor = MatchingDoctor(MatchingDoctorConfig)
            matching_doctor.diagnose()
            
            
if __name__ == '__main__':
    data_hospital2 = DataHospital2(DataHospitalConfig)
    data_hospital2.run()