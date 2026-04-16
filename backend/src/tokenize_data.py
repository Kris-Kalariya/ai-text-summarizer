from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-small')

def tokenize(data):
    inputs = tokenizer(
        data['dialogue'],
        padding="max_length",
        max_length=512,
        truncation=True
    )

    targets = tokenizer(
        data['summary'],
        padding="max_length",
        max_length=150,
        truncation=True
    )

    inputs['labels'] = targets['input_ids']
    return inputs