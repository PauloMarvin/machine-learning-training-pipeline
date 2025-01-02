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
        accuracy = accuracy_score(actual_labels_df, predictions_df)
        return float(accuracy)

    @staticmethod
    def compute_precision(
        predictions_df: pd.DataFrame,
        actual_labels_df: pd.DataFrame,
        average: str = "binary",
    ) -> float:
        precision = precision_score(actual_labels_df, predictions_df, average=average)
        return float(precision)

    @staticmethod
    def compute_recall(
        predictions_df: pd.DataFrame,
        actual_labels_df: pd.DataFrame,
        average: str = "binary",
    ) -> float:
        recall = recall_score(actual_labels_df, predictions_df, average=average)
        return float(recall)

    @staticmethod
    def compute_f1_score(
        predictions_df: pd.DataFrame,
        actual_labels_df: pd.DataFrame,
        average: str = "binary",
    ) -> float:
        f1 = f1_score(actual_labels_df, predictions_df, average=average)
        return float(f1)

    @staticmethod
    def compute_confusion_matrix(
        predictions_df: pd.DataFrame, actual_labels_df: pd.DataFrame
    ) -> pd.DataFrame:
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
        report = classification_report(actual_labels_df, predictions_df)
        return report
