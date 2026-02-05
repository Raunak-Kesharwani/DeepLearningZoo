from PIL import Image
import torch
from torchvision import transforms

MNIST_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

def preprocess_lenet_mnist(image: Image.Image) -> torch.Tensor:
    """
    Returns tensor of shape (1, 32, 32)
    """
    return MNIST_TRANSFORM(image)


CIFAR10_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
])

def preprocess_vgg_cifar10(image: Image.Image) -> torch.Tensor:
    """
    Returns tensor of shape (3, 32, 32)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    return CIFAR10_TRANSFORM(image)



CIFAR10_224_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
])

def preprocess_cifar10_224(image: Image.Image) -> torch.Tensor:
    """
    Returns tensor of shape (3, 224, 224)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    return CIFAR10_224_TRANSFORM(image)
