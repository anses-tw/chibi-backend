from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import httpx 
import base64
import urllib.parse

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

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    system_prompt = "你是一個AI繪圖專家。將中文想法翻譯為逗號分隔英文Prompt，加入chibi style, masterpiece等。只回傳英文。"
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json={"contents": [{"parts": [{"text": f"{system_prompt}\n想法：{req.chineseIdea}"}]}]})
        if resp.status_code != 200:
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

    google_url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predict?key={api_key}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 第一階段：先嘗試使用 Google 官方 API
        resp = await client.post(google_url, json={"instances": [{"prompt": req.prompt}], "parameters": {"sampleCount": 1}})
        
        if resp.status_code == 200:
            base64_img = resp.json()["predictions"][0]["bytesBase64Encoded"]
            return {"imageUrl": f"data:image/png;base64,{base64_img}"}
        elif resp.status_code == 429:
            raise HTTPException(status_code=429, detail="魔法額度滿載，請稍後再試！")
        else:
            print(f"Google 生圖拒絕 (轉用備用線路): {resp.text}")
            
            # 【終極救援機制】當 Google 鎖區時，自動切換至開源備用生圖 API
            try:
                fallback_prompt = urllib.parse.quote(req.prompt)
                fallback_url = f"https://image.pollinations.ai/prompt/{fallback_prompt}?width=512&height=512&nologo=true"
                
                fb_resp = await client.get(fallback_url)
                if fb_resp.status_code == 200:
                    # 將圖片轉換回前端看得懂的 Base64 格式
                    base64_img = base64.b64encode(fb_resp.content).decode('utf-8')
                    return {"imageUrl": f"data:image/png;base64,{base64_img}"}
                else:
                    raise HTTPException(status_code=400, detail="備用生圖線路也忙碌中，請稍後再試")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"生成圖片失敗: {str(e)}")