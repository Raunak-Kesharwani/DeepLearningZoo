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
