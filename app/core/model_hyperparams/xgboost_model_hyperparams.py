from pydantic import BaseModel


class XGBoostModelHyperparams(BaseModel):
    """
    Handles the storage and management of hyperparameters for an XGBoost model.
    This class is a schema definition using Pydantic's BaseModel to define and
    validate the hyperparameters required to configure an XGBoost model.

    This class is intended for use in scenarios where the user needs to configure
    the hyperparameters of an XGBoost model programmatically, ensure validity
    of input values, and manage default settings.

    :ivar max_depth: The maximum depth of a tree. Controls overfitting; larger
        values allow the model to capture more complex patterns but may result
        in overfitting.
    :type max_depth: int
    :ivar learning_rate: Boosting learning rate (also known as eta). Determines
        the contribution of each tree to the model; smaller values make the
        optimization process more stable at the cost of longer training time.
    :type learning_rate: float
    :ivar n_estimators: The number of boosting rounds or trees in the model.
        Controls the number of iterations of model training.
    :type n_estimators: int
    :ivar objective: The learning task objective. Defines the loss function to
        be minimized, such as "binary:logistic" for binary classification.
    :type objective: str
    :ivar booster: The booster type to use, such as "gbtree" (default) for tree-based
        models or "gblinear" for linear models.
    :type booster: str
    :ivar n_jobs: The number of parallel threads to use for model training. Controls
        the level of parallelism in training; -1 means using all processors.
    :type n_jobs: int
    """

    max_depth: int = 3
    learning_rate: float = 0.1
    n_estimators: int = 100
    objective: str = "binary:logistic"
    booster: str = "gbtree"
    n_jobs: int = 1
