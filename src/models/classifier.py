import torch
from torch import nn
from torchvision import models


def create_resnet18(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def unfreeze_layer4(model: nn.Module) -> None:
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True


def main() -> None:
    model = create_resnet18(num_classes=6, freeze_backbone=True)

    dummy_batch = torch.randn(4, 3, 224, 224)
    outputs = model(dummy_batch)

    print(model.__class__.__name__)
    print(f"Output shape: {outputs.shape}")
    print(f"Trainable parameters: {count_trainable_parameters(model)}")


if __name__ == "__main__":
    main()
