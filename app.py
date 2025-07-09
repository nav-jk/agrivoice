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

# ✅ Load Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"🔧 Using device: {device}")

asr_model_id = "openai/whisper-large-v3"
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    asr_model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True
).to(device)
asr_processor = AutoProcessor.from_pretrained(asr_model_id)

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

# ✅ Load Flan-T5 + LoRA
base_model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(base_model_name)
base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)

lora_model_path = "./flan_t5_base_agri_lora_gpu"
model = PeftModel.from_pretrained(base_model, lora_model_path)
model = model.to(device)
model.eval()

print("✅ Flan-T5 LoRA model ready")

# ✅ LoRA Inference
def generate_answer(instruction: str):
    print(f"📥 Instruction to LLM:\n{instruction}")
    input_ids = tokenizer(instruction, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)

    outputs = model.generate(
        input_ids=input_ids,
        max_length=128,
        do_sample=True,
        top_p=0.95,
        temperature=0.7
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"📤 LLM Response:\n{decoded}")
    return decoded

# ✅ /chat endpoint
@app.post("/chat/")
async def chat(file: UploadFile = File(...)):
    print("📡 Received audio file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    # Transcribe audio
    print("🔊 Transcribing...")
    asr_result = asr_pipe(audio_path)
    os.remove(audio_path)

    question = asr_result["text"].strip()
    print(f"🗣 Transcribed Text: {question}")

    if not question:
        return {"error": "Transcription failed or empty"}

    # Format the prompt
    instruction = f"State: MAHARASHTRA, Crop: Wheat, Category: 0, Query Type: 2\nFarmer's Question: {question}"

    # Generate LLM response
    response = generate_answer(instruction)

    return {
        "transcription": question,
        "response": response
    }
