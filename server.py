"""Jalankan: uvicorn server:app --reload  ->  buka http://127.0.0.1:8000"""
import hmac
import json
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import modelservice as ms

BASE = Path(__file__).parent
app = FastAPI(title="Prediksi Nilai Akhir Siswa")
SESSIONS: dict[str, str] = {}  # token -> username (hilang kalau server restart)


class Login(BaseModel):
    username: str
    password: str


def current_user(authorization: str = Header(None)) -> str:
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token not in SESSIONS:
        raise HTTPException(401, "Sesi habis, silakan masuk lagi")
    return SESSIONS[token]


@app.post("/api/login")
def login(body: Login):
    users = json.loads((BASE / "users.json").read_text("utf-8"))
    stored = users.get(body.username)
    if stored is None or not hmac.compare_digest(stored.encode(), body.password.encode()):
        raise HTTPException(401, "Username atau password salah")
    token = secrets.token_hex(16)
    SESSIONS[token] = body.username
    return {"token": token, "user": body.username}


@app.post("/api/logout")
def logout(authorization: str = Header(None)):
    SESSIONS.pop((authorization or "").removeprefix("Bearer ").strip(), None)
    return {"ok": True}


@app.get("/api/info")
def get_info(_: str = Depends(current_user)):
    return ms.info()


@app.post("/api/predict")
def get_predict(payload: dict, _: str = Depends(current_user)):
    try:
        clean = {k: float(v) for k, v in payload.items()}
    except (TypeError, ValueError):
        raise HTTPException(422, "Semua input harus berupa angka")
    grade, probs = ms.predict(clean)
    return {"grade": grade, "probs": probs}


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")