import typing
from collections import OrderedDict
from typing import Tuple
import joblib
import pandas as pd
from pydantic import BaseModel
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from flytekit.types.structured import StructuredDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Constants
TARGET_LABEL_INDEX = 8  # Index for the target label column ("class")

# Dataset columns schema
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
FEATURE_COLUMNS = {
    col: dtype for col, dtype in DATASET_COLUMNS.items() if col != "class"
}
CLASS_COLUMNS = {"class": int}


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def split_train_test_dataset(
    dataset_file: FlyteFile[typing.TypeVar("csv")], seed: int, test_split_ratio: float
) -> Tuple[StructuredDataset, StructuredDataset, StructuredDataset, StructuredDataset]:
    """
    Splits a dataset CSV file into train and test sets for features and labels.

    Args:
        dataset_file (FlyteFile): The CSV file containing the dataset.
        seed (int): Random state seed for reproducibility.
        test_split_ratio (float): Ratio of the test split.

    Returns:
        Tuple[StructuredDataset]: Training features, testing features, training labels, testing labels.
    """
    column_names = list(DATASET_COLUMNS.keys())
    df = pd.read_csv(dataset_file, names=column_names)

    # Extract features and labels
    features_df = df.iloc[:, :TARGET_LABEL_INDEX]
    labels_df = df.iloc[:, [TARGET_LABEL_INDEX]]

    # Use helper function to split data
    train_features, test_features, train_labels, test_labels = split_data(
        features_df, labels_df, seed, test_split_ratio
    )

    return (
        StructuredDataset(dataframe=train_features),
        StructuredDataset(dataframe=test_features),
        StructuredDataset(dataframe=train_labels),
        StructuredDataset(dataframe=test_labels),
    )


def split_data(
    features: pd.DataFrame, labels: pd.DataFrame, seed: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Helper method for splitting features and labels into training and testing datasets.

    Args:
        features (pd.DataFrame): Input features DataFrame.
        labels (pd.DataFrame): Input labels DataFrame.
        seed (int): Random seed for reproducibility.
        test_split_ratio (float): Train-test split ratio.

    Returns:
        Tuple[pd.DataFrame]: Training and testing datasets for features and labels.
    """
    return train_test_split(
        features, labels, test_size=test_split_ratio, random_state=seed
    )


# Model hyperparameter definitions
class XGBoostModelHyperparams(BaseModel):
    """
    These are the XGBoost hyperparameters available in the scikit-learn library.
    """

    max_depth: int = 3
    learning_rate: float = 0.1
    n_estimators: int = 100
    objective: str = "binary:logistic"
    booster: str = "gbtree"
    n_jobs: int = 1


# Model artifact and workflow outputs
MODELSER_JOBLIB = typing.TypeVar("joblib.dat")
model_file = typing.NamedTuple("Model", model=FlyteFile[MODELSER_JOBLIB])
workflow_outputs = typing.NamedTuple(
    "WorkflowOutputs", model=FlyteFile[MODELSER_JOBLIB], accuracy=float
)


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def train_model(
    features: StructuredDataset,
    labels: StructuredDataset,
    hyperparams: XGBoostModelHyperparams,
) -> model_file:
    """
    Trains an XGBoost classifier on the provided features and labels, and outputs the serialized model file.

    Args:
        features (StructuredDataset): Structured dataset of features for training.
        labels (StructuredDataset): Structured dataset of labels for training.
        hyperparams (XGBoostModelHyperparams): Hyperparameters for the XGBoost classifier.

    Returns:
        model_file: A named tuple containing the serialized model file.
    """
    features_df = features.open(dataframe_type=pd.DataFrame).all()
    labels_df = labels.open(dataframe_type=pd.DataFrame).all()

    # Initialize and fit the model
    model = XGBClassifier(
        n_jobs=hyperparams.n_jobs,
        max_depth=hyperparams.max_depth,
        n_estimators=hyperparams.n_estimators,
        booster=hyperparams.booster,
        objective=hyperparams.objective,
        learning_rate=hyperparams.learning_rate,
    )
    model.fit(features_df, labels_df)

    # Serialize the model
    model_filename = "model.joblib.dat"
    joblib.dump(model, model_filename)
    return (model_filename,)


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def make_predictions(
    features: StructuredDataset,
    model_file: FlyteFile[MODELSER_JOBLIB],
) -> StructuredDataset:
    """
    Predicts labels using the trained model and provided features.

    Args:
        features (StructuredDataset): Input features for prediction.
        model_file (FlyteFile): Serialized model file.

    Returns:
        StructuredDataset: Predictions as a structured dataset.
    """
    model = joblib.load(model_file)
    features_df = features.open(dataframe_type=pd.DataFrame).all()
    predictions = model.predict(features_df)

    # Return predictions as a DataFrame
    predictions_df = pd.DataFrame(predictions, columns=["class"], dtype="int64")
    return StructuredDataset(dataframe=predictions_df)


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def calculate_accuracy(
    predictions: StructuredDataset, actual_labels: StructuredDataset
) -> float:
    """
    Calculates the accuracy score of the predictions against the actual labels.

    Args:
        predictions (StructuredDataset): Predicted labels.
        actual_labels (StructuredDataset): Ground truth labels.

    Returns:
        float: Accuracy score as a floating-point number.
    """
    predictions_df = predictions.open(dataframe_type=pd.DataFrame).all()
    actual_labels_df = actual_labels.open(dataframe_type=pd.DataFrame).all()

    # Evaluate accuracy
    accuracy = accuracy_score(actual_labels_df, predictions_df)
    print(f"Accuracy: {accuracy:.2%}")
    return float(accuracy)


@workflow
def diabetes_xgboost_pipeline(
    dataset_file: FlyteFile[
        typing.TypeVar("csv")
    ] = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    test_split_ratio: float = 0.33,
    seed: int = 7,
) -> workflow_outputs:
    """
    Main Flyte workflow that trains an XGBoost model on the provided dataset, evaluates it, and returns the results.

    Args:
        dataset_file (FlyteFile): Dataset file containing the diabetes data.
        test_split_ratio (float): Ratio for test dataset split.
        seed (int): Random seed for reproducibility.

    Returns:
        workflow_outputs: Named tuple containing the serialized model and accuracy score.
    """
    train_features, test_features, train_labels, test_labels = split_train_test_dataset(
        dataset_file=dataset_file, seed=seed, test_split_ratio=test_split_ratio
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
    return trained_model.model, accuracy


if __name__ == "__main__":
    print("Running main pipeline...")
    print(diabetes_xgboost_pipeline())
