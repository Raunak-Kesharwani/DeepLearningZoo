MODELS = {
    "LeNet5 (MNIST)": {
        "type": "image",
        "model_path": "models/image/lenet_mnist/model.pt",
        "preprocess": "preprocess_lenet_mnist",
        "input_shape": (1, 32, 32),
        "classes": [str(i) for i in range(10)],
        "docs": {
            "info": "models/image/lenet_mnist/info.md",
            "learning": "models/image/lenet_mnist/learning.md"
        }
    }
}


MODELS["VGG_CNN (CIFAR-10)"] = {
    "type": "image",
    "model_path": "models/image/vgg_cnn_cifar10/model.pt",
    "preprocess": "preprocess_vgg_cifar10",
    "input_shape": (3, 32, 32),
    "classes": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ],
    "docs": {
        "info": "models/image/vgg_cnn_cifar10/info.md",
        "learning": "models/image/vgg_cnn_cifar10/learning.md"
    }
}


