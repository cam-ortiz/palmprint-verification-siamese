import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve
)

class PalmprintMetrics:
    def __init__(self, y_true, y_pred, y_scores=None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_scores = None if y_scores is None else np.array(y_scores)

    # ----------------------------
    # Basic Classification Metrics
    # ----------------------------
    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def precision(self):
        return precision_score(self.y_true, self.y_pred, zero_division=0)

    def recall(self):
        return recall_score(self.y_true, self.y_pred, zero_division=0)

    def f1_score(self):
        return f1_score(self.y_true, self.y_pred, zero_division=0)

    def confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    # ----------------------------
    # Biometric Metrics
    # ----------------------------
    def far_frr(self):
        cm = self.confusion_matrix()

        if cm.shape != (2, 2):
            return None, None
        
        TN, FP, FN, TP = cm.ravel()

        FAR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FRR = FN / (TP + FN) if (TP + FN) > 0 else 0

        return FAR, FRR

    def eer(self):
        if self.y_scores is None:
            return None

        fpr, tpr, _ = roc_curve(self.y_true, self.y_scores)
        fnr = 1 - tpr

        idx = np.nanargmin(np.abs(fpr - fnr))
        return fpr[idx]

    def tar_at_far(self, target_far=0.01):
        if self.y_scores is None:
            return None

        fpr, tpr, _ = roc_curve(self.y_true, self.y_scores)
        idx = np.argmin(np.abs(fpr - target_far))
        return tpr[idx]

    # ----------------------------
    # Report
    # ----------------------------
    def report(self):
        print("==== Palmprint Evaluation Metrics ====")
        print(f"Accuracy : {self.accuracy():.4f}")
        print(f"Precision: {self.precision():.4f}")
        print(f"Recall   : {self.recall():.4f}")
        print(f"F1-Score : {self.f1_score():.4f}")

        FAR, FRR = self.far_frr()
        if FAR is not None:
            print(f"FAR      : {FAR:.4f}")
            print(f"FRR      : {FRR:.4f}")

        eer = self.eer()
        if eer is not None:
            print(f"EER      : {eer:.4f}")

        tar = self.tar_at_far()
        if tar is not None:
            print(f"TAR@FAR=0.01: {tar:.4f}")
        
        print("\nConfusion Matrix:")
        print(self.confusion_matrix())
