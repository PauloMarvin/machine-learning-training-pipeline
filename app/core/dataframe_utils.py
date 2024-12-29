from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataFrameUtils:

    @staticmethod
    def split_data(
        features: pd.DataFrame, labels: pd.DataFrame, seed: int, test_split_ratio: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        return train_test_split(
            features, labels, test_size=test_split_ratio, random_state=seed
        )
