from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms.v2 import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, Dataset, DataLoader


class TransformDataset(Dataset):
    def __init__(self, dataset, transforms):
        super(TransformDataset, self).__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return self.transforms(x), y


def get_dataloaders(path_to_train_dataset, path_to_test_dataset, batch_size):
    dataset = ImageFolder(path_to_train_dataset)
    test_dataset = ImageFolder(path_to_test_dataset)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    # Трансформации для датасетов
    train_transforms = Compose([
        RandomHorizontalFlip(p=0.2),
        RandomVerticalFlip(p=0.2),
        RandomRotation([-5, 5], fill=255.),
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.5), (0.5))
    ])

    test_transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.5), (0.5))
    ])

    # Добавление трансформаций к датасетам
    train_dataset = TransformDataset(train_dataset, train_transforms)
    val_dataset = TransformDataset(val_dataset, test_transforms)
    test_dataset = TransformDataset(test_dataset, test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Проверка
    print("Количество изображений в train:", len(train_dataset))
    print("Количество изображений в val:", len(val_dataset))
    print("Список классов:", dataset.classes)
    return train_loader, val_loader, test_loader, dataset.classes
