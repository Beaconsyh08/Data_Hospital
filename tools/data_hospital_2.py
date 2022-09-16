from configs.config import DataHospitalConfig, EvaluateProcessorConfig, MissAnnoCheckerConfig, MatchingCheckerConfig, StatisticsManagerConfig
from src.data_hospital.evaluate_processor_lucas import EvaluateProcessorLucas
from src.data_hospital.evaluate_processor_qa import EvaluateProcessorQA
from src.data_hospital.matching_checker import MatchingChecker
from src.data_hospital.miss_anno_checker import MissAnnoChecker
from src.data_hospital.statistics_manager import StatisticsManager
from src.utils.logger import get_logger


logger = get_logger()

class DataHospital2():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.modules = cfg.MODULES
        
    def run(self, ) -> None:
        
        if self.cfg.ORIENTATION == "SIDE":
            logger.warning("Data Checking II Running")
            
            if "Evaluate" in self.modules:
                logger.warning("Evaluator Processor Running")
                evaluate_processor = EvaluateProcessorLucas(EvaluateProcessorConfig) if self.cfg.EVALUATOR == "LUCAS" else EvaluateProcessorQA(EvaluateProcessorConfig)
                evaluate_processor.diagnose()
                    
            if "MissAnno" in self.modules:
                logger.warning("Miss Annotation Checker Running")
                miss_anno_checker = MissAnnoChecker(MissAnnoCheckerConfig)
                miss_anno_checker.diagnose()
                
            if "Matching" in self.modules:
                logger.warning("2D-3D Matching Checker Running")
                matching_checker = MatchingChecker(MatchingCheckerConfig)
                matching_checker.diagnose()
                
            if "Statistics" in self.modules:
                logger.warning("Statistics Manager Running")
                statistics_manager = StatisticsManager(StatisticsManagerConfig)
                statistics_manager.diagnose()
            
        logger.warning("Data Checking II Completed")
            
            
if __name__ == '__main__':
    data_hospital2 = DataHospital2(DataHospitalConfig)
    data_hospital2.run()