from configs.config import DataHospitalConfig, DuplicatedCheckerConfig, StatisticsManagerConfig, LogicalCheckerConfig, CalibrationCheckerConfig, CoordinateConverterConfig, InferenceConfig
from src.data_hospital.coordinate_converter import CoordinateConverter
from src.data_hospital.calibration_checker import CalibrationChecker
from src.data_hospital.logical_checker import LogicalChecker
from src.data_hospital.duplicated_checker import DuplicatedChecker
from src.data_hospital.statistics_manager import StatisticsManager
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
            logger.critical("Duplicated Checker Diagnosing")
            duplicated_checker = DuplicatedChecker(DuplicatedCheckerConfig)
            duplicated_checker.self_diagnose()
        
        if "Logical" in self.modules:
            logger.critical("Logical Checker Diagnosing")
            logical_checker = LogicalChecker(LogicalCheckerConfig)
            logical_checker.diagnose()
            logical_checker.txt_for_reproejct()
            
        if "Calibration" in self.modules:
            logger.critical("Calibration Checker Diagnosing")
            calibration_checker = CalibrationChecker(CalibrationCheckerConfig)
            calibration_checker.diagnose()
        
        if DataHospitalConfig.COOR == "Lidar":
            if "CoordinateConverter" in self.modules:
                logger.critical("Coor Trans Checker Diagnosing")
                coordinate_converter = CoordinateConverter(CoordinateConverterConfig)
                coordinate_converter.diagnose()
        elif DataHospitalConfig.COOR == "Vehicle":
            if "CoordinateConverter" in self.modules:
                os.makedirs(CoordinateConverterConfig.OUTPUT_DIR, exist_ok=True)
                os.system("cp %s %s" % (CoordinateConverterConfig.INPUT_PATH, CoordinateConverterConfig.OUTPUT_PATH))
            
        
        logger.critical("Data Hospital I Completed")
        

if __name__ == '__main__':
    data_hospital = DataHospital(DataHospitalConfig)
    data_hospital.run()
    

    
    # logger.critical("Stats Checker Diagnosing")
    # statistics_manager = StatisticsManager(StatisticsManagerConfig)
    # statistics_manager.diagnose()
    
    
