import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class MetricUtils:

    @staticmethod
    def compute_accuracy(
        predictions_df: pd.DataFrame, actual_labels_df: pd.DataFrame
    ) -> float:
        """
        Computes the accuracy of predictions against the actual labels.

        :param predictions_df: Pandas DataFrame containing the predicted labels.
        :param actual_labels_df: Pandas DataFrame containing the actual labels.
        :return: Accuracy as a float, representing the proportion of correct predictions.
        """
        accuracy = accuracy_score(actual_labels_df, predictions_df)
        return float(accuracy)

    @staticmethod
    def compute_precision(
        predictions_df: pd.DataFrame,
        actual_labels_df: pd.DataFrame,
        average: str = "binary",
    ) -> float:
        """
        Computes precision score for classification predictions.

        :param predictions_df: Pandas DataFrame containing the predicted labels.
        :param actual_labels_df: Pandas DataFrame containing the actual labels.
        :param average: Averaging method for multiclass problems ("binary", "micro", "macro", "weighted").
                        Default is "binary".
        :return: Precision score as a float.
        """
        precision = precision_score(actual_labels_df, predictions_df, average=average)
        return float(precision)

    @staticmethod
    def compute_recall(
        predictions_df: pd.DataFrame,
        actual_labels_df: pd.DataFrame,
        average: str = "binary",
    ) -> float:
        """
        Computes recall score for classification predictions.

        :param predictions_df: Pandas DataFrame containing the predicted labels.
        :param actual_labels_df: Pandas DataFrame containing the actual labels.
        :param average: Averaging method for multiclass problems ("binary", "micro", "macro", "weighted").
                        Default is "binary".
        :return: Recall score as a float.
        """
        recall = recall_score(actual_labels_df, predictions_df, average=average)
        return float(recall)

    @staticmethod
    def compute_f1_score(
        predictions_df: pd.DataFrame,
        actual_labels_df: pd.DataFrame,
        average: str = "binary",
    ) -> float:
        """
        Computes F1 score for classification predictions.

        :param predictions_df: Pandas DataFrame containing the predicted labels.
        :param actual_labels_df: Pandas DataFrame containing the actual labels.
        :param average: Averaging method for multiclass problems ("binary", "micro", "macro", "weighted").
                        Default is "binary".
        :return: F1 score as a float.
        """
        f1 = f1_score(actual_labels_df, predictions_df, average=average)
        return float(f1)

    @staticmethod
    def compute_confusion_matrix(
        predictions_df: pd.DataFrame, actual_labels_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes the confusion matrix for the classification predictions.

        :param predictions_df: Pandas DataFrame containing the predicted labels.
        :param actual_labels_df: Pandas DataFrame containing the actual labels.
        :return: Confusion matrix as a Pandas DataFrame.
        """
        matrix = confusion_matrix(actual_labels_df, predictions_df)
        return pd.DataFrame(
            matrix,
            columns=["Predicted Negative", "Predicted Positive"],
            index=["Actual Negative", "Actual Positive"],
        )

    @staticmethod
    def classification_report(
        predictions_df: pd.DataFrame, actual_labels_df: pd.DataFrame
    ) -> str:
        """
        Generates a detailed classification report.

        :param predictions_df: Pandas DataFrame containing the predicted labels.
        :param actual_labels_df: Pandas DataFrame containing the actual labels.
        :return: A string containing the classification report.
        """
        report = classification_report(actual_labels_df, predictions_df)
        return report
