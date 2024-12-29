from typing import List, Literal, NamedTuple, Tuple

import joblib
import pandas as pd
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from flytekit.types.structured import StructuredDataset
from xgboost import XGBClassifier

from app.core.dataframe_utils import DataFrameUtils

# DATASET_COLUMNS = OrderedDict(
#     {
#         "#preg": int,
#         "pgc_2h": int,
#         "diastolic_bp": int,
#         "tricep_skin_fold_mm": int,
#         "serum_insulin_2h": int,
#         "bmi": float,
#         "diabetes_pedigree": float,
#         "age": int,
#         "class": int,
#     }
# )


@task(cache_version="1.0", cache=False, limits=Resources(mem="200Mi"))
def split_train_test_dataset(
    dataset_file: FlyteFile[Literal["csv"]],
    seed: int,
    test_split_ratio: float,
    column_names: List[str],
    target_label_column_index: int,
) -> Tuple[StructuredDataset, StructuredDataset, StructuredDataset, StructuredDataset]:

    # column_names = list(DATASET_COLUMNS.keys())
    print(f"Column names: {column_names}")
    df = pd.read_csv(dataset_file, names=column_names)

    features_df = df.iloc[:, :target_label_column_index]
    labels_df = df.iloc[:, [target_label_column_index]]

    train_features, test_features, train_labels, test_labels = (
        DataFrameUtils.split_data(features_df, labels_df, seed, test_split_ratio)
    )

    return (
        StructuredDataset(dataframe=train_features),
        StructuredDataset(dataframe=test_features),
        StructuredDataset(dataframe=train_labels),
        StructuredDataset(dataframe=test_labels),
    )


MODELSER_JOBLIB = Literal["joblib.dat"]
model_file_joblib = NamedTuple("Model", model=FlyteFile[MODELSER_JOBLIB])
workflow_outputs = NamedTuple(
    "WorkflowOutputs", model=FlyteFile[MODELSER_JOBLIB], accuracy=float
)

from app.core.model_hyperparams.xgboost_model_hyperparams import XGBoostModelHyperparams


@task(cache_version="1.0", cache=False, limits=Resources(mem="200Mi"))
def train_model(
    features: StructuredDataset,
    labels: StructuredDataset,
    hyperparams: XGBoostModelHyperparams,
) -> model_file_joblib:
    features_df = features.open(dataframe_type=pd.DataFrame).all()
    labels_df = labels.open(dataframe_type=pd.DataFrame).all()

    model = XGBClassifier(
        n_jobs=hyperparams.n_jobs,
        max_depth=hyperparams.max_depth,
        n_estimators=hyperparams.n_estimators,
        booster=hyperparams.booster,
        objective=hyperparams.objective,
        learning_rate=hyperparams.learning_rate,
    )
    model.fit(features_df, labels_df)

    model_filename = "../models/pima_indians_basic_workflow.joblib.dat"
    joblib.dump(model, model_filename)

    return (model_filename,)


@task(cache_version="1.0", cache=False, limits=Resources(mem="200Mi"))
def make_predictions(
    features: StructuredDataset,
    model_file: FlyteFile[Literal["joblib.dat"]],
) -> StructuredDataset:
    model = joblib.load(model_file)
    features_df = features.open(dataframe_type=pd.DataFrame).all()
    predictions = model.predict(features_df)

    predictions_df = pd.DataFrame(predictions, columns=["class"], dtype="int64")
    return StructuredDataset(dataframe=predictions_df)


@task(cache_version="1.0", cache=False, limits=Resources(mem="200Mi"))
def calculate_accuracy(
    predictions: StructuredDataset, actual_labels: StructuredDataset
) -> float:
    predictions_df = predictions.open(dataframe_type=pd.DataFrame).all()
    actual_labels_df = actual_labels.open(dataframe_type=pd.DataFrame).all()
    from app.core.metric_utils import MetricUtils

    accuracy = MetricUtils.compute_accuracy(predictions_df, actual_labels_df)
    print(f"Accuracy: {accuracy:.2%}")
    return float(accuracy)


COLUMNS_NAMES = [
    "#preg",
    "pgc_2h",
    "diastolic_bp",
    "tricep_skin_fold_mm",
    "serum_insulin_2h",
    "bmi",
    "diabetes_pedigree",
    "age",
    "class",
]

TARGET_LABEL_INDEX = 8


@workflow
def diabetes_xgboost_pipeline(
    column_names: List[str],
    target_label_column_index: int,
    dataset_file: FlyteFile[
        Literal["csv"]
    ] = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    test_split_ratio: float = 0.20,
    seed: int = 7,
) -> workflow_outputs:
    train_features, test_features, train_labels, test_labels = split_train_test_dataset(
        dataset_file=dataset_file,
        seed=seed,
        test_split_ratio=test_split_ratio,
        column_names=column_names,
        target_label_column_index=target_label_column_index,
    )
    trained_model = train_model(
        features=train_features,
        labels=train_labels,
        hyperparams=XGBoostModelHyperparams(max_depth=4),
    )
    predictions = make_predictions(
        features=test_features, model_file=trained_model.model
    )
    accuracy = calculate_accuracy(predictions=predictions, actual_labels=test_labels)
    return (trained_model.model, accuracy)


if __name__ == "__main__":
    print("Running main pipeline...")
    print(
        diabetes_xgboost_pipeline(
            column_names=COLUMNS_NAMES, target_label_column_index=TARGET_LABEL_INDEX
        )
    )
