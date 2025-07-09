import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from peft import PeftModel
from fastapi import FastAPI, UploadFile, File
import tempfile
import os

app = FastAPI()

# ✅ Load Whisper model for transcription
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Whisper setup
whisper_model_id = "openai/whisper-large-v3"
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)
processor = AutoProcessor.from_pretrained(whisper_model_id)

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
    task="translate",  # Translates to English if needed
)
print(f"✅ Whisper model loaded on {device}")

# ✅ Load LLM (Flan-T5 + LoRA)
base_model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(base_model_name)
base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)

lora_model_path = "./flan_t5_base_agri_lora_gpu"
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()
model.to(device)

def generate_answer(transcribed_text: str) -> str:
    # Customize instruction template here if needed
    instruction = f"State: MAHARASHTRA, Crop: Wheat, Category: 0, Query Type: 2\nFarmer's Question: {transcribed_text}"
    input_ids = tokenizer(instruction, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
    with torch.no_grad():
        outputs = model.base_model.generate(input_ids, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ Combined API
@app.post("/chat")
async def chat_with_audio(file: UploadFile = File(...)):
    # Step 1: Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Step 2: Transcribe audio
    try:
        transcription_result = asr_pipe(tmp_path)
        transcribed_text = transcription_result["text"]
    finally:
        os.remove(tmp_path)

    # Step 3: Generate LLM response
    response = generate_answer(transcribed_text)

    return {
        "transcription": transcribed_text,
        "response": response
    }
