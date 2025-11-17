import torch.nn as nn
import torchvision.models as models


class PharmDetector(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super(PharmDetector, self).__init__()
        # Используем предобученную ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Замораживаем часть слоев для трансферного обучения
        if freeze_backbone:
            # Замораживаем все слои кроме последних двух
            for name, param in self.backbone.named_parameters():
                if not name.startswith('layer4') and not name.startswith('fc'):
                    param.requires_grad = False
        # Заменяем последний полносвязный слой под наше количество классов
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
