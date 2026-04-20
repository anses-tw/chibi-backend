from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import httpx 

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 【終極防呆】自動清除 API Key 頭尾不小心複製到的空白鍵或換行
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

@app.post("/api/translate")
async def translate_prompt(req: TranslateRequest):
    api_key = API_KEYS.get(req.groupId)
    if not api_key:
        raise HTTPException(status_code=400, detail="無效組別或尚未設定金鑰")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    system_prompt = "你是一個AI繪圖專家。將中文想法翻譯為逗號分隔英文Prompt，加入chibi style, masterpiece等。只回傳英文。"
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json={"contents": [{"parts": [{"text": f"{system_prompt}\n想法：{req.chineseIdea}"}]}]})
        if resp.status_code != 200:
            # 故意回傳 400 來區分錯誤，避免跟路徑 404 搞混
            print(f"Google 翻譯報錯: {resp.text}")
            raise HTTPException(status_code=400, detail=f"Google API 拒絕 (代碼 {resp.status_code})")
        return {"englishPrompt": resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()}

class GenerateRequest(BaseModel):
    groupId: int
    prompt: str

@app.post("/api/generate")
async def generate_image(req: GenerateRequest):
    api_key = API_KEYS.get(req.groupId)
    if not api_key:
        raise HTTPException(status_code=400, detail="無效組別或尚未設定金鑰")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predict?key={api_key}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json={"instances": [{"prompt": req.prompt}], "parameters": {"sampleCount": 1}})
        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail="魔法額度滿載，請稍後再試！")
        if resp.status_code != 200:
            print(f"Google 生圖報錯: {resp.text}")
            raise HTTPException(status_code=400, detail=f"Google API 拒絕 (代碼 {resp.status_code})")
            
        base64_img = resp.json()["predictions"][0]["bytesBase64Encoded"]
        return {"imageUrl": f"data:image/png;base64,{base64_img}"}