import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import Callable, Optional

class TestDataset(datasets.CIFAR10):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = transforms.ToTensor(), target_transform: Optional[Callable] = None, download: bool = False):
        super(TestDataset, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int):
        image, label = super(TestDataset, self).__getitem__(index)
        return image, label
