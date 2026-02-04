import torch
import torch.nn.functional as F

def infer_image_classifier(model, image_tensor, class_names):
    """
    image_tensor: (1, 32, 32)
    """
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))   # (1, 10)
        probs = F.softmax(logits, dim=1).squeeze(0)

    pred_idx = probs.argmax().item()

    return {
        "prediction": class_names[pred_idx],
        "probabilities": {
            class_names[i]: probs[i].item()
            for i in range(len(class_names))
        }
    }
