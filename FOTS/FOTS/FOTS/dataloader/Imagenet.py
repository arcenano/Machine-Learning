import torch.utils.data # Dataloader
import torchvision.transforms as transforms # Image transformations
import torchvision.datasets as datasets # Imagenet Dataset

class ImageNet:
    """
    Reference
    https://github.com/MrtnMndt/Deep_Openset_Recognition_through_Uncertainty/blob/master/lib/Datasets/datasets.py

    Other (Sequential Loader - Alternative to pytorch)
    https://github.com/BayesWatch/sequential-imagenet-dataloader/blob/master/examples/imagenet/main.py

    ImageNet dataset consisting of a large amount (more than a million images) images with varying
    resolution for 1000 different classes.
    Typically the smaller side of the images is rescaled to 256 and quadratic random 224x224 crops
    are taken. Dataloader implemented with torchvision.datasets.ImageNet.
    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, config):
        self.num_classes = config.classes
        self.dataset = config.data_dir

        print("getting transforms")

        self.train_transforms, self.val_transforms = self.__get_transforms(config.patch_size)


        print("getting dataset")

        self.trainset, self.valset = self.get_dataset()

        print("getting loader")

        self.train_loader, self.val_loader = self.get_dataset_loader(config.batch_size, config.workers, is_gpu, config.shuffle)

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.Resize(patch_size + 32),
            transforms.RandomResizedCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(patch_size + 32),
            transforms.CenterCrop(patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.ImageNet to load dataset.
        Downloads dataset if doesn't exist already.
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.ImageNet(self.dataset, split='train', transform=self.train_transforms,
                                     target_transform=None)
        valset = datasets.ImageNet(self.dataset, split='val', transform=self.val_transforms,   
                                   target_transform=None)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu,shuffle):
        """
        Defines the dataset loader for wrapped dataset
        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader