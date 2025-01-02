from pydantic import BaseModel


class XGBoostModelHyperparams(BaseModel):
    max_depth: int = 3
    learning_rate: float = 0.1
    n_estimators: int = 100
    objective: str = "binary:logistic"
    booster: str = "gbtree"
    n_jobs: int = 1
