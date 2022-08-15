from configs.config import DataHospital2Config, EvaluateConfig
from src.data_hospital.evaluate_doctor import EvaluateDoctor
from src.utils.logger import get_logger


logger = get_logger()

class DataHospital2():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.modules = cfg.MODULES
        
    def run(self, ) -> None:
        
        logger.critical("Data Hospital II Running")
        
        if "Evaluate" in self.modules:
            evaluate_doctor = EvaluateDoctor(EvaluateConfig)
            evaluate_doctor.diagnose()
        
if __name__ == '__main__':
    data_hospital2 = DataHospital2(DataHospital2Config)
    data_hospital2.run()
    
