from collections import OrderedDict
from typing import Literal, NamedTuple, Tuple

import joblib
import pandas as pd
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from flytekit.types.structured import StructuredDataset
from pydantic import BaseModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Constants
TARGET_LABEL_INDEX = 8

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


def split_data(
    features: pd.DataFrame, labels: pd.DataFrame, seed: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the provided dataset into training and testing subsets based on the specified
    test split ratio and random seed. This function is designed to work with datasets
    represented as Pandas DataFrame objects for both features and labels, ensuring
    compatibility with a wide range of data preprocessing workflows.

    :param features: A Pandas DataFrame containing the feature data of the dataset, where
        each row represents an instance and each column represents a feature.
    :param labels: A Pandas DataFrame containing the label data of the dataset, where each
        row corresponds to the label(s) of the corresponding instance in the features.
    :param seed: An integer used to seed the random number generator for reproducibility
        when splitting the dataset into training and testing subsets.
    :param test_split_ratio: A float in the range (0, 1) that specifies the proportion of
        the dataset to be allocated to the testing set.
    :return: A tuple containing four Pandas DataFrames: the training feature set, the
        testing feature set, the training label set, and the testing label set, in that
        order.
    """
    return train_test_split(
        features, labels, test_size=test_split_ratio, random_state=seed
    )


@task(cache_version="1.0", cache=False, limits=Resources(mem="200Mi"))
def split_train_test_dataset(
    dataset_file: FlyteFile[Literal["csv"]], seed: int, test_split_ratio: float
) -> Tuple[StructuredDataset, StructuredDataset, StructuredDataset, StructuredDataset]:
    """
    Splits a dataset into training and testing subsets for both features and labels. The dataset is
    provided as a file, and this function utilizes randomness controlled by a seed value to split
    the data based on the specified ratio. The function outputs structured datasets for training
    features, test features, training labels, and test labels respectively.

    :param dataset_file: The input dataset file in CSV format containing feature columns and a
        label column. It is expected to match the structure defined in the DATASET_COLUMNS dictionary.
    :param seed: The random seed used to control the reproducibility of the train-test split process.
    :param test_split_ratio: A float value representing the proportion of the dataset to include in
        the test split. Must be a value between 0 and 1.
    :return: A tuple containing four StructuredDataset objects:
        - Training features
        - Test features
        - Training labels
        - Test labels
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


# Model artifact and workflow outputs
# MODELSER_JOBLIB = typing.TypeVar("joblib.dat")
# model_file = typing.NamedTuple("Model", model=FlyteFile[MODELSER_JOBLIB])
model_file = NamedTuple("Model", model=FlyteFile[Literal["joblib.dat"]])


@task(cache_version="1.0", cache=False, limits=Resources(mem="200Mi"))
def train_model(
    features: StructuredDataset,
    labels: StructuredDataset,
    hyperparams: XGBoostModelHyperparams,
) -> FlyteFile[Literal["joblib.dat"]]:
    """
    Train a machine learning model using features and labels from structured datasets.
    This function initializes an XGBoost classifier with given hyperparameters, fits
    the model, and serializes it to a file. It is optimized for resource constraints
    and caching.

    :param features: Input feature dataset in a structured format.
    :type features: StructuredDataset
    :param labels: Target label dataset in a structured format.
    :type labels: StructuredDataset
    :param hyperparams: Configuration object containing hyperparameters for the
        XGBoost classifier.
    :type hyperparams: XGBoostModelHyperparams
    :return: The file path to the serialized trained model.
    :rtype: model_file
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

    return FlyteFile[Literal["joblib.dat"]](path=model_filename)


@task(cache_version="1.0", cache=False, limits=Resources(mem="200Mi"))
def make_predictions(
    features: StructuredDataset,
    model_file: FlyteFile[Literal["joblib.dat"]],
) -> StructuredDataset:
    """
    Makes predictions based on the input features using a pre-trained model. This function
    loads a machine learning model from the provided serialized model file, processes the
    input features, and generates predictions. The predictions are returned as a
    structured dataset in DataFrame format.

    :param features: Input structured dataset containing the features required for
        generating predictions. The dataset is expected to represent rows of data for
        prediction purposes.
    :type features: StructuredDataset
    :param model_file: Serialized file containing a pre-trained machine learning model.
        The model must be stored in `joblib` format and compatible with the features.
    :type model_file: FlyteFile[MODELSER_JOBLIB]
    :return: A structured dataset containing predicted classes as a DataFrame with
        a single column labeled "class".
    :rtype: StructuredDataset
    """
    model = joblib.load(model_file)
    features_df = features.open(dataframe_type=pd.DataFrame).all()
    predictions = model.predict(features_df)

    # Return predictions as a DataFrame
    predictions_df = pd.DataFrame(predictions, columns=["class"], dtype="int64")
    return StructuredDataset(dataframe=predictions_df)


@task(cache_version="1.0", cache=False, limits=Resources(mem="200Mi"))
def calculate_accuracy(
    predictions: StructuredDataset, actual_labels: StructuredDataset
) -> float:
    """
    Calculate the accuracy of predictions against the actual labels using a structured dataset.

    This function compares the predicted labels to the actual labels provided,
    and computes the accuracy as a float value representing the percentage of correct predictions.

    :param predictions: A structured dataset containing the predicted labels. The type of data should align
        with pandas DataFrame standards for further processing.
    :param actual_labels: A structured dataset containing the actual labels to compare against. The type
        of data should align with pandas DataFrame standards for further processing.
    :return: The accuracy score as a float, representing the proportion of correct predictions over the
        total labels.
    """
    predictions_df = predictions.open(dataframe_type=pd.DataFrame).all()
    actual_labels_df = actual_labels.open(dataframe_type=pd.DataFrame).all()

    # Evaluate accuracy
    accuracy = accuracy_score(actual_labels_df, predictions_df)
    print(f"Accuracy: {accuracy:.2%}")
    return float(accuracy)


workflow_outputs = NamedTuple(
    "WorkflowOutputs", model=FlyteFile[Literal["csv"]], accuracy=float
)


@workflow
def diabetes_xgboost_pipeline(
    dataset_file: FlyteFile[
        Literal["csv"]
    ] = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    test_split_ratio: float = 0.20,
    seed: int = 7,
) -> workflow_outputs:
    """
    The function `diabetes_xgboost_pipeline` implements a machine learning pipeline for
    the diabetes dataset using the XGBoost model. This pipeline involves splitting the
    dataset into training and testing sets, training an XGBoost model on the training
    data, making predictions on the testing data, and calculating the accuracy of
    the predictions. The pipeline can be adjusted using configurable parameters for
    the dataset, test split ratio, and random seed.

    :param dataset_file: A FlyteFile object representing the CSV file containing the dataset.
    :param test_split_ratio: A float value representing the proportion of the dataset to be used for testing.
    :param seed: An integer representing the seed value for random generator to ensure reproducibility.
    :return: A tuple where the first element is the trained model file, and the second is the computed accuracy of the model.
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
    return workflow_outputs(model=trained_model.model, accuracy=accuracy)


if __name__ == "__main__":
    print("Running main pipeline...")
    print(diabetes_xgboost_pipeline())
