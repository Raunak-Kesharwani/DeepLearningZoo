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





def generate_answer_seq2seq(
    model,
    tokenizer,
    question: str,
    context: str,
    max_len: int = 40,
):
    """
    Generic inference for Seq2Seq QA (with or without attention)

    model: TorchScript Seq2Seq model
    tokenizer: tokenizers.Tokenizer
    question: question string
    context: context string
    """

    model.eval()

    # ---- Special tokens ----
    pad_id = tokenizer.token_to_id("<pad>")
    sos_id = tokenizer.token_to_id("<sos>")
    eos_id = tokenizer.token_to_id("<eos>")

    # ---- Build encoder input ----
    # IMPORTANT: must match training format
    enc_text = question + " " + context
    enc_ids = tokenizer.encode(enc_text).ids

    enc_x = torch.tensor(enc_ids, dtype=torch.long).unsqueeze(0)

    # ---- Decoder starts with <sos> ----
    dec_ids = [sos_id]

    with torch.no_grad():
        for _ in range(max_len):
            dec_x = torch.tensor(dec_ids, dtype=torch.long).unsqueeze(0)

            # Forward pass
            logits = model(enc_x, dec_x)

            # Take last timestep
            next_token_logits = logits[0, -1]
            next_token_id = next_token_logits.argmax().item()

            if next_token_id == eos_id:
                break

            dec_ids.append(next_token_id)

    # Remove <sos>
    answer_ids = dec_ids[1:]

    return tokenizer.decode(answer_ids)
