import torch
from torch import nn, optim

from src.dataloaders import get_dataloaders
from src.learning import run_train
from src.model import PharmDetector

# Создаем даталоадеры
batch_size = 256
path_to_train_dataset = "../dataset/ogyeiv2/train"
path_to_test_dataset = "../dataset/ogyeiv2/test"
train_loader, val_loader, test_loader, all_classes = get_dataloaders(path_to_train_dataset,
                                                                     path_to_test_dataset,
                                                                     batch_size)

# Создаем модель
model = PharmDetector(num_classes=len(all_classes), freeze_backbone=True)

# Настройка гиперпараметров
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f'Обучаем на {device}')

# Запуск обучения с тестированием и выводом метрик
run_train(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader,
          val_loader=val_loader, test_loader=test_loader, class_names=all_classes, epochs=10, device=device)
