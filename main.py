from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
import httpx 
import base64
import urllib.parse
import random

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_clean_key(group_id):
    key = os.getenv(f"GEMINI_API_KEY_{group_id}")
    return key.strip() if key else None

API_KEYS = {
    1: get_clean_key(1),
    2: get_clean_key(2),
    3: get_clean_key(3),
    4: get_clean_key(4),
    5: get_clean_key(5),
}

class TranslateRequest(BaseModel):
    groupId: int
    chineseIdea: str
    imageBase64: Optional[str] = None 

@app.post("/api/translate")
async def translate_prompt(req: TranslateRequest):
    api_key = API_KEYS.get(req.groupId)
    if not api_key:
        raise HTTPException(status_code=400, detail="無效組別或尚未設定金鑰")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    parts = []
    system_prompt = "你是一個專業的 AI 提示詞工程師。"
    
    if req.imageBase64:
        system_prompt += "請仔細觀察這張照片中人物的特徵（例如：性別、髮型、髮色、動作手勢、穿著）。將這些『照片特徵』與使用者的『中文想法』完美融合。"
        parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": req.imageBase64
            }
        })
    else:
        system_prompt += "將使用者的『中文想法』轉換為英文。"

    system_prompt += f"\n請寫出一串用於 AI 繪圖的英文 Prompt（逗號分隔）。必須包含 chibi style, cute anime character, masterpiece, highly detailed。如果使用者有特別指定特定服裝，請覆蓋照片原本的服裝。只回傳純英文 prompt，不要任何多餘廢話對話。\n使用者的額外想法：{req.chineseIdea}"

    parts.append({"text": system_prompt})
    
    payload = {"contents": [{"parts": parts}]}
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                return {"englishPrompt": resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()}
            else:
                print(f"Google 視覺分析報錯: {resp.text}")
                raise Exception("Google API Error")
    except Exception as e:
        print(f"Google 翻譯/視覺分析失敗，啟用【終極不斷線備用方案】: {e}")
        # 【終極防呆機制】當 Google 突然傲嬌拒絕時，系統會自動給出一組安全咒語，保證畫面流程絕對不會卡住！
        safe_idea = req.chineseIdea if req.chineseIdea else "cute student"
        fallback_safe_prompt = f"chibi style, cute anime character, {safe_idea}, highly detailed, masterpiece"
        return {"englishPrompt": fallback_safe_prompt}

class GenerateRequest(BaseModel):
    groupId: int
    prompt: str

@app.post("/api/generate")
async def generate_image(req: GenerateRequest):
    api_key = API_KEYS.get(req.groupId)
    if not api_key:
        raise HTTPException(status_code=400, detail="無效組別或尚未設定金鑰")

    google_url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predict?key={api_key}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 嘗試第一防線：Google 官方生圖
            resp = await client.post(google_url, json={"instances": [{"prompt": req.prompt}], "parameters": {"sampleCount": 1}})
            
            if resp.status_code == 200:
                base64_img = resp.json()["predictions"][0]["bytesBase64Encoded"]
                return {"imageUrl": f"data:image/png;base64,{base64_img}"}
            else:
                print(f"Google 生圖報錯 (準備切換備用線路): {resp.text}")
                raise Exception("Google Imagen API Error")
    except Exception as e:
        print(f"Google 拒絕生圖，【自動切換至 Pollinations 備用伺服器】: {e}")
        try:
            # 嘗試第二防線：開源生圖伺服器 (將咒語長度安全裁切，防止網址過長)
            safe_prompt = req.prompt[:500] 
            fallback_prompt = urllib.parse.quote(safe_prompt)
            seed_number = random.randint(1, 1000000)
            fallback_url = f"https://image.pollinations.ai/prompt/{fallback_prompt}?width=512&height=512&nologo=true&seed={seed_number}"
            
            async with httpx.AsyncClient(timeout=30.0) as fb_client:
                fb_resp = await fb_client.get(fallback_url)
                if fb_resp.status_code == 200:
                    base64_img = base64.b64encode(fb_resp.content).decode('utf-8')
                    return {"imageUrl": f"data:image/png;base64,{base64_img}"}
                else:
                    raise HTTPException(status_code=400, detail="備用伺服器繁忙，請稍後再試一次！")
        except Exception as fb_e:
            raise HTTPException(status_code=400, detail=f"所有生圖線路皆忙碌中，請重新嘗試: {str(fb_e)}")