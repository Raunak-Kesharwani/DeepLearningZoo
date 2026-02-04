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

