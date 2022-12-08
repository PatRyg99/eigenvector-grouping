import torch
from torchmetrics.classification.confusion_matrix import (
    ConfusionMatrix as TorchConfusionMatrix,
)


class ConfusionMatrix(TorchConfusionMatrix):
    def __call__(self, pred: torch.Tensor, label: torch.Tensor):
        values = super().__call__(pred.flatten(), label.flatten())
        self.TN = values[0, 0]
        self.FP = values[0, 1]
        self.FN = values[1, 0]
        self.TP = values[1, 1]

    def precision(self):
        return torch.nan_to_num(self.TP / (self.TP + self.FP))

    def recall(self):
        return torch.nan_to_num(self.TP / (self.TP + self.FN))

    def accuracy(self):
        return torch.nan_to_num(
            (self.TP + self.TN) / (self.TP + self.FN + self.TN + self.FP)
        )

    def f1_score(self):
        p = self.precision()
        r = self.recall()

        return torch.nan_to_num(2 * p * r / (p + r))

    def specifity(self):
        return torch.nan_to_num(self.TN / (self.TN + self.FP))
