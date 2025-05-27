from ..base_policy import BasePolicy
import torchvision
import torch
import torch.nn.functional as F


class TestPolicy(BasePolicy):
    def __init__(self, num_classes: int = 10):
        super(TestPolicy, self).__init__()
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def compute_loss(self, batch):
        images = batch[0]
        labels = batch[1]
        preds = self.model(images)  # (batch_size, num_classes)
        labels = labels.long()
        loss = torch.nn.functional.cross_entropy(preds, labels)  # (batch_size,)
        return loss
    
    def predict_action(self, image: torch.Tensor):
        return self.model(image)