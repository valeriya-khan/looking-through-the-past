from torchvision import datasets, transforms
from data.manipulate import UnNormalize
from torch.utils.data import Dataset
from PIL import Image

class MiniDataset(Dataset):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        self.train = train
        self.trsf = transform
        self.target_transform = target_transform
        if train is True:
            with open("/raid/NFS_SHARE/home/valeriya.khan/continual-learning/store/datasets/MINI/mini_cl_train.csv", 'r') as f:
                lines = list(map(lambda x: (x).replace("\n", "").split(","), f.readlines()))
        else:
            with open("/raid/NFS_SHARE/home/valeriya.khan/continual-learning/store/datasets/MINI/mini_cl_test.csv", 'r') as f:
                lines = list(map(lambda x: (x).replace("\n", "").split(","), f.readlines()))
        
        self.images, self.labels = zip(*lines)
        self.labels = [int(i) for i in self.labels]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # if self.use_path:
        image = self.trsf(pil_loader(self.images[idx]))
        # else:
        #     image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.target_transform(self.labels[idx])

        return image, label
    

def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

# specify available data-sets.
AVAILABLE_DATASETS = {
    'MNIST': datasets.MNIST,
    'CIFAR100': datasets.CIFAR100,
    'CIFAR10': datasets.CIFAR10,
    'CIFAR50': datasets.CIFAR100,
    "MINI": MiniDataset,
    'TINY': datasets.ImageFolder

}

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'MNIST': [
        transforms.ToTensor(),
    ],
    'MNIST32': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'CIFAR10': [
        transforms.ToTensor(),
    ],
    'CIFAR50': [
        transforms.ToTensor(),
    ],
    'CIFAR100': [
        transforms.ToTensor(),
    ],
    'MINI': [
        transforms.ToTensor(),
    ],
    'TINY': [
        transforms.ToTensor(),
    ],
    'CIFAR10_norm': [
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ],
    'CIFAR50_norm': [
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    ],
    'CIFAR100_norm': [
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    ],
    'MINI_norm': [
        transforms.Normalize(mean=[0.47313006, 0.44905752, 0.40378186], std=[0.27292014, 0.26559181, 0.27953038]),
    ],
    'TINY_norm': [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    'CIFAR10_denorm': UnNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    'CIFAR50_denorm': UnNormalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    'CIFAR100_denorm': UnNormalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    'MINI_denorm': UnNormalize(mean=[0.47313006, 0.44905752, 0.40378186], std=[0.27292014, 0.26559181, 0.27953038]),
    'TINY_denorm': UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'augment_from_tensor': [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ],
    'augment': [
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
    ],
    'augment_mini': [
        transforms.RandomCrop(84, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
    ],
    'augment_tiny': [
        transforms.Resize((32,32)),
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
    ]
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'MNIST': {'size': 28, 'channels': 1, 'classes': 10},
    'MNIST32': {'size': 32, 'channels': 1, 'classes': 10},
    'CIFAR10': {'size': 32, 'channels': 3, 'classes': 10},
    'CIFAR100': {'size': 32, 'channels': 3, 'classes': 100},
    'CIFAR50': {'size': 32, 'channels': 3, 'classes': 100},
    'MINI': {'size': 84, 'channels': 3, 'classes': 100},
    'TINY': {'size': 32, 'channels': 3, 'classes': 200},
}
