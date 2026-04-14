from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from src.preprocess import clean_data

# Load model
model = T5ForConditionalGeneration.from_pretrained('models/saved_summery_model')
tokenizer = T5Tokenizer.from_pretrained('models/saved_summery_model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize_dialogue(dialogue):
    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True
    ).to(device)

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


# Test
if __name__ == "__main__":
    test_dialogue = """
    A: Hey, are we still meeting today?
    B: Yes, at 5 PM.
    A: Great, see you then.
    """

    print("Summary:", summarize_dialogue(test_dialogue))