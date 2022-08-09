from src.stats.stats import Stats
import pandas as pd


class SingleStats(Stats):
    def __init__(self, df: pd.DataFrame = None, df_path: str = None, cfg: dict = None) -> None:
        super(SingleStats, self).__init__(df=df, df_path=df_path, cfg=cfg)
        