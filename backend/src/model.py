from transformers import T5ForConditionalGeneration
import torch

def load_model():
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    return model, device