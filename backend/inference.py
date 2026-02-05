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



def infer_lstm_timeseries(model, sequence_tensor):
    """
    sequence_tensor: torch.Tensor of shape (1, 20, 1)
    Returns a float prediction (next timestep)
    """
    model.eval()

    with torch.no_grad():
        output = model(sequence_tensor)   # (1, 1)

    return output.item()



def generate_text_lm(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens=100,
    temperature=1.0
):
    model.eval()

    # Encode prompt
    token_ids = tokenizer.encode(prompt).ids
    token_ids = token_ids[-60:]  # truncate if needed

    for _ in range(max_new_tokens):
        x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(x)

        next_logits = logits[0, -1] / temperature
        probs = F.softmax(next_logits, dim=0)

        next_token = torch.multinomial(probs, 1).item()
        token_ids.append(next_token)

        if len(token_ids) > 60:
            token_ids = token_ids[-60:]

    return tokenizer.decode(token_ids)
