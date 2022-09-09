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
            logger.critical("Data Hospital II Running")
            logger.error("Activated Modules: %s" % ', '.join(map(str, self.modules)))
            
            if "Evaluate" in self.modules:
                evaluate_processor = EvaluateProcessorLucas(EvaluateProcessorConfig) if self.cfg.EVALUATOR == "LUCAS" else EvaluateProcessorQA(EvaluateProcessorConfig)
                evaluate_processor.diagnose()
                    
            if "MissAnno" in self.modules:
                miss_anno_checker = MissAnnoChecker(MissAnnoCheckerConfig)
                miss_anno_checker.diagnose()
                
            if "Matching" in self.modules:
                matching_checker = MatchingChecker(MatchingCheckerConfig)
                matching_checker.diagnose()
                
            if "Statistics" in self.modules:
                statistics_manager = StatisticsManager(StatisticsManagerConfig)
                statistics_manager.diagnose()
            
        logger.critical("Data Hospital II Completed")
            
            
if __name__ == '__main__':
    data_hospital2 = DataHospital2(DataHospitalConfig)
    data_hospital2.run()