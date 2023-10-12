from monai.metrics import DiceMetric
class MeanDiceMetric:
    def __init__(self):
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.best_score = 0.0  # Initialize with a low value

    def reset(self):
        self.dice_metric.reset()

    def compute(self, y_pred, y):
        self.dice_metric(y_pred=y_pred, y=y)

    def get_mean_dice(self):
        return self.dice_metric.aggregate().item()

    def update_best_score(self):
        current_score = self.get_mean_dice()
        if current_score > self.best_score:
            self.best_score = current_score

