from torchvision import transforms
import numpy as np
import torch

# Returns a list of transformations when called

class GetTransforms():
    '''Returns a list of transformations when type as requested amongst train/test
       Transforms('train') = list of transforms to apply on training data
       Transforms('test') = list of transforms to apply on testing data'''

    def __init__(self):
        pass

    def trainparams(self):
        # train_transformations = [
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
        #     transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        #     transforms.RandomRotation((-7.0, 7.0), resample=False, expand=False, center=None, fill=(0.491, 0.482, 0.446)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))]

        train_transformations = [ #resises the image so it can be perfect for our model.
            transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
            transforms.RandomRotation(10),     #Rotates the image to a specified angel
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
            transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)) #Normalize all the images
            ]
        # train_transformations = [
        #     torchvision.transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        #     transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
        # ]

        return train_transformations

    def testparams(self):
        test_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
        ]
        return test_transforms

def Get_Mean_Dev(dataset):

    train_data = dataset.data
    (train_data).shape
    # train_data = trainset.transform(train_data)
    train_data = train_data/255
    print(np.mean(train_data, axis = (0,1,2)))
    print(np.std(train_data, axis = (0,1,2)))

    train_tensor = torch.from_numpy(train_data)

    print(f'Mean of dataset {torch.mean(train_tensor)}')
    print(f'Mean of dataset {torch.std(train_tensor)}')
    # print(np.mean(trainset.data[0], axis = 2))
