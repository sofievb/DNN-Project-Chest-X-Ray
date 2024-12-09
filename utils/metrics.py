import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from prettytable import PrettyTable

class Metrics:
    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def calculate_precision(y_true, y_pred, average="weighted"):
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def calculate_recall(y_true, y_pred, average="weighted"):
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def calculate_f1_score(y_true, y_pred, average="weighted"):
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def calculate_confusion_matrix(y_true, y_pred, labels=None):
        return confusion_matrix(y_true, y_pred, labels=labels)

    @staticmethod
    def calculate_all(y_true, y_pred, average="macro", multi_label=False):
        table = PrettyTable(["Metric", "Value"])
        if multi_label:
            # Threshold predictions for multi-label classification
            y_pred_bin = (y_pred > 0.5).astype(int)
            table.add_row(["Accuracy", Metrics.calculate_accuracy(y_true, y_pred_bin)])
            table.add_row(["Precision", Metrics.calculate_precision(y_true, y_pred_bin, average=average)])
            table.add_row(["Recall", Metrics.calculate_recall(y_true, y_pred_bin, average=average)])
            table.add_row(["F1 Score", Metrics.calculate_f1_score(y_true, y_pred_bin, average=average)])
        else:
            table.add_row(["Accuracy", Metrics.calculate_accuracy(y_true, y_pred)])
            table.add_row(["Precision", Metrics.calculate_precision(y_true, y_pred, average=average)])
            table.add_row(["Recall", Metrics.calculate_recall(y_true, y_pred, average=average)])
            table.add_row(["F1 Score", Metrics.calculate_f1_score(y_true, y_pred, average=average)])

        return table
