from fastapi import FastAPI, UploadFile, File
import torch
import tempfile
import os
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from peft import PeftModel

app = FastAPI()

# ✅ Load Whisper model for transcription
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Whisper components
asr_model_id = "openai/whisper-large-v3"
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    asr_model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True
).to(device)
asr_processor = AutoProcessor.from_pretrained(asr_model_id)

# Add translation task
forced_decoder_ids = asr_processor.get_decoder_prompt_ids(language="en", task="transcribe")

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=asr_model,
    tokenizer=asr_processor.tokenizer,
    feature_extractor=asr_processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"forced_decoder_ids": forced_decoder_ids}
)
print("✅ Whisper model ready")

# ✅ Load Flan-T5 + LoRA model
base_model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(base_model_name)
base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
lora_model_path = "./flan_t5_base_agri_lora_gpu"
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval().to(device)
print("✅ Flan-T5 LoRA model ready")

# ✅ Function to query the LoRA model
def generate_answer(instruction: str):
    input_ids = tokenizer(instruction, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
    outputs = model.base_model.generate(input_ids, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ /chat endpoint
@app.post("/chat/")
async def chat(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    # Transcribe
    asr_result = asr_pipe(audio_path)
    os.remove(audio_path)

    question = asr_result["text"]
    print(f"🗣 Transcribed Question: {question}")

    # Format instruction
    instruction = f"State: MAHARASHTRA, Crop: Wheat, Category: 0, Query Type: 2\nFarmer's Question: {question}"
    response = generate_answer(instruction)

    return {
        "transcription": question,
        "response": response
    }
