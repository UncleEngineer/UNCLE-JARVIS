# pip install sounddevice scipy openai gTTS playsound==1.2.2 requests

import os
import time
import base64
import requests
import sounddevice as sd
from scipy.io import wavfile
from openai import OpenAI
from gtts import gTTS
from playsound import playsound
from k import API_KEY
# ==========================================
# 1. ตั้งค่าระบบและ API (โปรดใส่ Key ของคุณให้ถูกต้อง)
# ==========================================
OPENROUTER_API_KEY = API_KEY  # เปลี่ยนเป็น OpenRouter API Key ของคุณ
MODEL_NAME = "nousresearch/hermes-3-llama-3.1-405b"  # หรือใช้ "google/gemini-2.5-flash" เพื่อความเร็ว

# เรียกใช้ Client สำหรับ LLM Chat
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SAMPLE_RATE = 44100
DURATION = 10  # ระยะเวลาอัดเสียง (วินาที)
AUDIO_INPUT_FILE = "user_command.wav"
AUDIO_OUTPUT_FILE = "jarvis_response.mp3"

# ==========================================
# 2. ฟังก์ชันอัดเสียงจากไมโครโฟน
# ==========================================
def record_audio():
    print(f"\n🎙️ JARVIS กำลังฟัง... (พูดได้เลย {DURATION} วินาที)")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # รอจนกว่าจะอัดเสียงเสร็จสมบูรณ์
    wavfile.write(AUDIO_INPUT_FILE, SAMPLE_RATE, recording)
    print("💾 บันทึกเสียงลงเครื่องเรียบร้อยแล้ว...")

# ==========================================
# 3. ฟังก์ชันแปลงเสียงเป็นข้อความ (STT) รองรับ OpenRouter
# ==========================================
def speech_to_text():
    print("🤖 กำลังแปลงเสียงเป็นข้อความ (ผ่าน Base64 JSON)...")
    
    # อ่านไฟล์เสียงแล้วเข้ารหัสเป็น Base64 string ตามข้อกำหนดของ OpenRouter
    with open(AUDIO_INPUT_FILE, "rb") as audio_file:
        audio_bytes = audio_file.read()
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
    
    url = "https://openrouter.ai/api/v1/audio/transcriptions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openai/whisper-large-v3",
        "input_audio": {
            "data": base64_audio,
            "format": "wav"
        },
        "language": "th"  # บังคับฟังและแกะภาษาไทย
    }
    
    # ส่งข้อมูลไปยัง OpenRouter Endpoint
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result_json = response.json()
        return result_json.get("text", "")
    else:
        raise Exception(f"OpenRouter STT Error: {response.status_code} - {response.text}")

# ==========================================
# 4. ฟังก์ชันส่งข้อความให้ AI คิดหาคำตอบ (LLM Brain)
# ==========================================
def ask_jarvis_brain(user_text):
    print(f"👤 คุณพูดว่า: '{user_text}'")
    print(f"🧠 กำลังส่งให้ {MODEL_NAME} ประมวลผลความคิด...")
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system", 
                "content": "คุณคือ JARVIS ผู้ช่วยอัจฉริยะในห้องทำงานของ Tony Stark ตอบกลับสั้นกระชับ เป็นกันเอง สุภาพ และมีไหวพริบ โดยต้องตอบเป็นภาษาไทยเท่านั้น"
            },
            {"role": "user", "content": user_text}
        ]
    )
    return response.choices[0].message.content

# ==========================================
# 5. ฟังก์ชันเปลี่ยนข้อความเป็นเสียงพูด (TTS) ด้วย playsound
# ==========================================
def text_to_speech(ai_response_text):
    print(f"🤖 JARVIS คิดคำตอบ: '{ai_response_text}'")
    print("🔊 กำลังสังเคราะห์เสียงเพื่อตอบกลับ...")
    
    # แปลงข้อความตัวอักษรเป็นไฟล์เสียงภาษาไทย
    tts = gTTS(text=ai_response_text, lang='th')
    tts.save(AUDIO_OUTPUT_FILE)
    
    # สั่งเล่นเสียงผ่านโปรแกรมเครื่องเล่นเสียง (บล็อกโปรแกรมหลักไว้จนกว่าเสียงจะเล่นจบ)
    playsound(AUDIO_OUTPUT_FILE)

# ==========================================
# 🏃‍♂️ ลูปการรันระบบหลัก (Main Pipeline)
# ==========================================
if __name__ == "__main__":
    try:
        # ขั้นตอนที่ 1: อัดเสียงของคุณผ่านไมโครโฟน
        record_audio()
        
        # ขั้นตอนที่ 2: แปลงเสียงที่บันทึกไว้ให้เป็นข้อความ
        user_text = speech_to_text()
        
        if not user_text or user_text.strip() == "":
            print("❌ ระบบไม่ได้ยินเสียงพูด หรือไฟล์เสียงไม่มีข้อมูลตัวอักษร")
        else:
            # ขั้นตอนที่ 3: ส่งตัวอักษรให้สมองกล AI ประมวลผลคำตอบ
            ai_reply = ask_jarvis_brain(user_text)
            
            # ขั้นตอนที่ 4: สังเคราะห์คำตอบของ AI ออกมาเป็นเสียงพูดทางลำโพง
            text_to_speech(ai_reply)
            
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในระบบ: {e}")