from fastapi import FastAPI, Request
from pydanticim import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates        # UI
from fastapi.responces import HTMLResponse
from fastapi.staticfiles import StaticFiles


# initialize you fastapi app
app = FastAPI(title="Text Summarization App", description="Text Summarization using T5", version="1.0")

# model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./saved_summery_model')
tokenizer = T5Tokenizer.from_pretrained('./saved_summery_model')

# device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

# templating
templates = Jinja2Templates(directory="./frontend")

# Input schema for dialogue => string

class DialogueInput(BaseModel):
    dialogue: str


# clean function
def clean_data(text):
    text = re.sub(r"\r\n", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

# summarization function
def summarize_dialogue(dialogue : str) -> str:
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



# API endpoints

@app.post('/summarize/')
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {'summary': summary}

@app.get('/', response_class = HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})