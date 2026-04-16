import pandas as pd
from transformers import Trainer, TrainingArguments
from backend.src.preprocess import clean_data
from backend.src.tokenize_data import tokenize
from backend.src.model import load_model

# Load data
train_data = pd.read_csv('data/samsum-train.csv')
val_data = pd.read_csv('data/samsum-validation.csv')

# Sampling
train_data = train_data.sample(n=4000, random_state=42).reset_index(drop=True)
val_data = val_data.sample(n=500, random_state=42).reset_index(drop=True)

# Cleaning
train_data['dialogue'] = train_data['dialogue'].apply(clean_data)
train_data['summary'] = train_data['summary'].apply(clean_data)

val_data['dialogue'] = val_data['dialogue'].apply(clean_data)
val_data['summary'] = val_data['summary'].apply(clean_data)

# Tokenization
train_dataset = train_data.apply(tokenize, axis=1).tolist()
val_dataset = val_data.apply(tokenize, axis=1).tolist()

# Load model
model, device = load_model()

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    weight_decay=0.01,
    warmup_steps=200
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train
trainer.train()

# Save model
model.save_pretrained('models/saved_summery_model')