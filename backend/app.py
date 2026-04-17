from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Initialize app
app = FastAPI(
    title="Text Summarization App",
    description="Text Summarization using T5",
    version="1.0"
)

# Load model
model = T5ForConditionalGeneration.from_pretrained('./models/saved_summery_model')
tokenizer = T5Tokenizer.from_pretrained('./models/saved_summery_model')

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Templates
templates = Jinja2Templates(directory="frontend")

# Static files (optional but recommended)
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Input schema
class DialogueInput(BaseModel):
    dialogue: str

# Clean function
def clean_data(text):
    text = re.sub(r"\r\n", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

# Summarization
def summarize_dialogue(dialogue: str) -> str:
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

# API
@app.post('/summarize/')
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {'summary': summary}

# UI route
@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})