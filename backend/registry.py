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

MODELS["Resnet_Tuned(CIFAR-10)"] = {
    "type": "image",
    "model_path": "models/image/resnet_cifar10_224/model.pt",
    "preprocess": "preprocess_cifar10_224",
    "input_shape": (3, 224, 224),
    "classes": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ],
    "docs": {
        "info": "models/image/resnet_cifar10_224/info.md",
        "learning": "models/image/resnet_cifar10_224/learning.md"
    }
}

MODELS["LSTM Sine Predictor"] = {
    "type": "timeseries",
    "model_path": "models/timeseries/model.pt",
    "input_length": 20,
    "input_shape": (20, 1),
    "output": "next_value",
    "docs": {
        "info": "models/timeseries/info.md",
        "learning": "models/timeseries/learning.md"
    }
}
