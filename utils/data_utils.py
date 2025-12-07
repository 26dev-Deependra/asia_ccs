import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy


def get_dataloader(config, train=True):
    transform = transforms.Compose([
        transforms.Resize(
            (config['model']['image_size'], config['model']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if train:
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
    else:
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['training']['batch_size'], shuffle=True)
    return loader


class Attacker:
    def __init__(self, config):
        self.config = config
        self.target_label = config['attack']['target_label']
        self.trigger_size = config['attack']['trigger_size']
        self.poison_ratio = config['attack']['poison_ratio']

    def poison_batch(self, images, labels):
        """
        Applies a pixel pattern trigger to a subset of the batch
        and changes labels to target.
        """
        poisoned_images = images.clone()
        poisoned_labels = labels.clone()

        batch_size = images.shape[0]
        num_poison = int(batch_size * self.poison_ratio)

        # Select indices to poison
        indices = np.random.choice(batch_size, num_poison, replace=False)

        for idx in indices:
            # Add White Pixel Trigger (Bottom Right)
            # Shape: [C, H, W]
            poisoned_images[idx, :, -self.trigger_size:, -
                            self.trigger_size:] = 1.0
            poisoned_labels[idx] = self.target_label

        return poisoned_images, poisoned_labels

    def is_active(self, current_round):
        return self.config['attack']['active'] and current_round >= self.config['attack']['start_round']


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total
