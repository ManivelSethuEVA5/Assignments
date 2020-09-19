# Code for downloading the data
from torchvision import datasets
from torchvision import transforms

from Assignments.Models.Model_Transformation import *

# from transformations import *

transformations = GetTransforms()
train_transforms = transforms.Compose(transformations.trainparams())
test_transforms = transforms.Compose(transformations.testparams())

print(train_transforms)

class Get_MNISTTrainData():
    def __init__(self, dir_name:str):
        self.dirname = dir_name

    def download_train_data(self):
        return datasets.MNIST('./data', train=True, download=True, transform=train_transforms)

    def download_test_data(self):
        return datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

class GetCIFAR10_TrainData():
    def __init__(self, dir_name:str):
        self.dirname = dir_name

    def download_train_data(self):
        return datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)

    def download_test_data(self):
        return datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)
