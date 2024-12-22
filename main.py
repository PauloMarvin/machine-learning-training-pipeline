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
    Splits a dataset into training and testing subsets for both features and labels. This
    function reads the provided dataset file, extracts feature and label columns, and
    divides the data into training and testing sets based on the specified test split ratio
    and random seed. The function ensures reproducibility through the provided seed value.

    :param dataset_file: Path to the dataset file in CSV format to be read as input.
    :param seed: Random seed for reproducibility when splitting the data.
    :param test_split_ratio: Proportion of the dataset to be used as the testing set.
    :return: A tuple containing four separate structured datasets:
        - Training features dataset.
        - Testing features dataset.
        - Training labels dataset.
        - Testing labels dataset.
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
    Splits the dataset into training and testing sets.

    This function takes the features and labels of a dataset,
    along with a random seed and the test split ratio, and splits the
    data into training and testing subsets. It ensures reproducibility of
    the split using the provided seed and allows for control over the
    proportion of the dataset allocated for testing using the test_split_ratio parameter.
    The split is performed such that the features and labels are partitioned
    consistently.

    :param features: The input features of the dataset to be split.
    :param labels: The corresponding labels of the dataset to be split.
    :param seed: An integer seed value for ensuring reproducibility of the data split.
    :param test_split_ratio: A float representing the proportion of the dataset
        to be allocated to the testing set.
    :return: A tuple containing four DataFrames:
        (X_train, X_test, y_train, y_test) where X_train and y_train correspond to
        the training data and labels respectively, and X_test and y_test correspond
        to the testing data and labels respectively.
    """
    return train_test_split(
        features, labels, test_size=test_split_ratio, random_state=seed
    )


# Model hyperparameter definitions
class XGBoostModelHyperparams(BaseModel):
    """
    Represents the hyperparameters for an XGBoost model configuration.

    This class is a data model used to specify the set of hyperparameters for an
    XGBoost model. It is primarily utilized to configure the model behavior,
    allowing customization of attributes like maximum tree depth, learning rate,
    number of boosting rounds, and more. This class is useful for applications
    involving supervised machine learning with XGBoost where hyperparameter tuning
    is necessary.

    :ivar max_depth: The maximum depth for each tree in the XGBoost model.
    :type max_depth: int
    :ivar learning_rate: Step size shrinkage used to prevent overfitting.
    :type learning_rate: float
    :ivar n_estimators: The number of boosting rounds (trees) to be built.
    :type n_estimators: int
    :ivar objective: The learning objective for the model (e.g., regression or classification tasks).
    :type objective: str
    :ivar booster: Specifies the type of booster to use in the model (e.g., 'gbtree', 'gblinear').
    :type booster: str
    :ivar n_jobs: The number of parallel threads used to run XGBoost.
    :type n_jobs: int
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
    Train a machine learning model using XGBoost with the supplied features, labels, and hyperparameters.
    The training process includes initializing an XGBoost classifier, fitting the model with input data,
    and serializing the trained model to a file for future use.

    :param features: Input dataset containing the features to be used for training the model.
    :type features: StructuredDataset
    :param labels: Input dataset containing the corresponding labels for the training features.
    :type labels: StructuredDataset
    :param hyperparams: Hyperparameters for the XGBoost model specifying configuration such as
        the number of estimators, maximum depth, learning rate, and other model-specific parameters.
    :type hyperparams: XGBoostModelHyperparams
    :return: Tuple containing the filepath of the serialized model file.
    :rtype: tuple
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
    This function makes predictions using a pre-trained model. It takes a dataset of
    features and a serialized model file, loads the model, and applies it to the
    input features to generate predictions. The predictions are then returned in a
    structured dataset format, encapsulated as a DataFrame.

    :param features: A structured dataset containing the features to be used as input
        for the model. This dataset is expected to be loaded as a pandas DataFrame.
    :type features: StructuredDataset
    :param model_file: A FlyteFile containing the serialized model. The model file should
        be in the MODELSER_JOBLIB format and compatible with `joblib.load`.
    :type model_file: FlyteFile[MODELSER_JOBLIB]
    :return: A structured dataset containing the predictions as a pandas DataFrame.
        Each row represents a prediction for the corresponding row in the input features
        dataset and includes a single column "class" with the predicted class labels.
    :rtype: StructuredDataset
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
    Calculate the accuracy of model predictions against actual labels.

    This function compares the predictions provided in a structured dataset with
    the actual labels in another structured dataset, calculating the accuracy
    score using a standard evaluation metric. The result is returned as a
    floating-point number representing the accuracy percentage.

    :param predictions: A structured dataset containing the model predictions.
        The dataset is expected to be compatible with a DataFrame format.
    :param actual_labels: A structured dataset containing the ground truth
        actual labels. The dataset is expected to be compatible with a
        DataFrame format.
    :return: The computed accuracy score as a floating-point number,
        representing the agreement between the predictions and
        actual labels as a percentage.
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
    Diabetes prediction pipeline using the XGBoost model. The function implements the
    training and testing process of a diabetes prediction model using a dataset. It performs
    data splitting, model training, predictions, and evaluates the accuracy of predictions.

    :param dataset_file: Input dataset file, expected to be in CSV format.
        The dataset is used to train and test the diabetes prediction model.
    :param test_split_ratio: Ratio of test data split from the dataset. Should
        be a float value between 0 and 1.
    :param seed: Random seed used for reproducibility during data splitting.
    :return: A tuple consisting of the trained model file path and the
        accuracy of the model on the test set.
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
