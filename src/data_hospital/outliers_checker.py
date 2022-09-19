from src.utils.logger import get_logger


class OutlierChecker():
    def __init__(self, cfg: dict) -> None:
        self.logger = get_logger()
        self.cfg = cfg       
        
        
    def diagnose() -> None:
        pass