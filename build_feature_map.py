import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from data import HuggingFaceCIFAR10Dataset, build_feature_transform, load_cifar10_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="features_map.npy")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--weights",
        type=str,
        default="imagenet",
        choices=("imagenet", "none"),
    )
    return parser.parse_args()


def build_encoder(weights_name: str):
    weights = ResNet18_Weights.IMAGENET1K_V1 if weights_name == "imagenet" else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    return model


def extract_features(model: nn.Module, dataloader: DataLoader, device: torch.device):
    all_features = []

    with torch.inference_mode():
        for images, _ in dataloader:
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, full_dataset = load_cifar10_dataset()
    feature_transform = build_feature_transform(image_size=args.image_size)
    dataset = HuggingFaceCIFAR10Dataset(full_dataset, transform=feature_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_encoder(args.weights).to(device)
    features = extract_features(model, dataloader, device)
    np.save(args.output, features)

    print(f"Saved feature map with shape {features.shape} to {args.output}")


if __name__ == "__main__":
    main()
