import typing
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import joblib
import pandas as pd
from dataclasses_json import dataclass_json
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from flytekit.types.structured import StructuredDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATASET_COLUMNS = OrderedDict(
    {
        "#preg": int,
        "pgc_2h": int,
        "diastolic_bp": int,
        "tricep_skin_fold_mm": int,
        "serum_insulin_2h": int,
        "bmi": float,
        "diabetes_pedigree": float,
        "age": int,
        "class": int,
    }
)

FEATURE_COLUMNS = OrderedDict(
    {k: v for k, v in DATASET_COLUMNS.items() if k != "class"}
)
CLASSES_COLUMNS = OrderedDict({"class": int})


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def split_traintest_dataset(
    dataset: FlyteFile[typing.TypeVar("csv")], seed: int, test_split_ratio: float
) -> Tuple[
    StructuredDataset,
    StructuredDataset,
    StructuredDataset,
    StructuredDataset,
]:
    """
    Retrieves the training dataset from the given blob location and then splits
    it using the split ratio and returns the result. The last column is assumed
    to be the class, and all other columns 0-8 are features.

    The data is returned as structured datasets.
    """
    column_names = [k for k in DATASET_COLUMNS.keys()]
    df = pd.read_csv(dataset, names=column_names)

    # Select all features
    x = df[column_names[:8]]
    # Select only the classes
    y = df[[column_names[-1]]]

    # split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_split_ratio, random_state=seed
    )

    return (
        StructuredDataset(dataframe=x_train),
        StructuredDataset(dataframe=x_test),
        StructuredDataset(dataframe=y_train),
        StructuredDataset(dataframe=y_test),
    )


MODELSER_JOBLIB = typing.TypeVar("joblib.dat")


@dataclass_json
@dataclass
class XGBoostModelHyperparams:
    """
    These are the xgboost hyper parameters available in scikit-learn library.
    """

    max_depth: int = 3
    learning_rate: float = 0.1
    n_estimators: int = 100
    objective: str = "binary:logistic"
    booster: str = "gbtree"
    n_jobs: int = 1


model_file = typing.NamedTuple("Model", model=FlyteFile[MODELSER_JOBLIB])
workflow_outputs = typing.NamedTuple(
    "WorkflowOutputs", model=FlyteFile[MODELSER_JOBLIB], accuracy=float
)


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def fit(
    x: StructuredDataset,
    y: StructuredDataset,
    hyperparams: XGBoostModelHyperparams,
) -> model_file:
    """
    This function takes the given input features and their corresponding classes to train a XGBClassifier.
    """
    x_df = x.open().all()
    y_df = y.open().all()

    # fit model on training data
    m = XGBClassifier(
        n_jobs=hyperparams.n_jobs,
        max_depth=hyperparams.max_depth,
        n_estimators=hyperparams.n_estimators,
        booster=hyperparams.booster,
        objective=hyperparams.objective,
        learning_rate=hyperparams.learning_rate,
    )
    m.fit(x_df, y_df)

    # Serialize model as a file
    fname = "model.joblib.dat"
    joblib.dump(m, fname)
    return (fname,)


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def predict(
    x: StructuredDataset,
    model_ser: FlyteFile[MODELSER_JOBLIB],
) -> StructuredDataset:
    """
    Given a trained model (serialized using joblib) and features, this method returns predictions.
    """
    model = joblib.load(model_ser)
    x_df = x.open().all()
    y_pred = model.predict(x_df)

    y_pred_df = pd.DataFrame(y_pred, columns=["class"], dtype="int64")
    return StructuredDataset(dataframe=y_pred_df)


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def score(predictions: StructuredDataset, y: StructuredDataset) -> float:
    """
    Compares the predictions with the actuals and returns the accuracy score.
    """
    pred_df = predictions.open().all()
    y_df = y.open().all()
    # evaluate predictions
    acc = accuracy_score(y_df, pred_df)
    print("Accuracy: %.2f%%" % (acc * 100.0))
    return float(acc)


@workflow
def diabetes_xgboost_model(
    dataset: FlyteFile[
        typing.TypeVar("csv")
    ] = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    test_split_ratio: float = 0.33,
    seed: int = 7,
) -> workflow_outputs:
    """
    This pipeline trains an XGBoost model for any given dataset that matches the schema as specified in
    https://github.com/jbrownlee/Datasets/blob/master/pima-indians-diabetes.names.
    """
    x_train, x_test, y_train, y_test = split_traintest_dataset(
        dataset=dataset, seed=seed, test_split_ratio=test_split_ratio
    )
    model = fit(
        x=x_train,
        y=y_train,
        hyperparams=XGBoostModelHyperparams(max_depth=4),
    )
    predictions = predict(x=x_test, model_ser=model.model)
    return model.model, score(predictions=predictions, y=y_test)


if __name__ == "__main__":
    print(f"Running {__file__} main...")
    print(diabetes_xgboost_model())
