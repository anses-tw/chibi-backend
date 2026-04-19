from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import httpx 

load_dotenv()
app = FastAPI()

# 允許前端連線到後端 (CORS 設定)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEYS = {
    1: os.getenv("GEMINI_API_KEY_1"),
    2: os.getenv("GEMINI_API_KEY_2"),
    3: os.getenv("GEMINI_API_KEY_3"),
    4: os.getenv("GEMINI_API_KEY_4"),
    5: os.getenv("GEMINI_API_KEY_5"),
}

class TranslateRequest(BaseModel):
    groupId: int
    chineseIdea: str

@app.post("/api/translate")
async def translate_prompt(req: TranslateRequest):
    if req.groupId not in API_KEYS or not API_KEYS[req.groupId]:
        raise HTTPException(status_code=400, detail="無效組別")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEYS[req.groupId]}"
    system_prompt = "你是一個AI繪圖專家。將中文想法翻譯為逗號分隔英文Prompt，加入chibi style, masterpiece等。只回傳英文。"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json={"contents": [{"parts": [{"text": f"{system_prompt}\n想法：{req.chineseIdea}"}]}]})
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="翻譯服務異常")
        return {"englishPrompt": resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()}

class GenerateRequest(BaseModel):
    groupId: int
    prompt: str

@app.post("/api/generate")
async def generate_image(req: GenerateRequest):
    if req.groupId not in API_KEYS or not API_KEYS[req.groupId]:
        raise HTTPException(status_code=400, detail="無效組別")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predict?key={API_KEYS[req.groupId]}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json={"instances": [{"prompt": req.prompt}], "parameters": {"sampleCount": 1}})
        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail="魔法額度滿載，請稍後再試！")
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="生圖失敗")
            
        base64_img = resp.json()["predictions"][0]["bytesBase64Encoded"]
        return {"imageUrl": f"data:image/png;base64,{base64_img}"}