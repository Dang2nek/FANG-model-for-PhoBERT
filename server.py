"""
server.py - Flask API server cho hệ thống phát hiện BLHĐ
Chạy lệnh: python server.py
Sau đó mở file blhd_warning.html trên trình duyệt
"""

import torch
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Cấu hình ────────────────────────────────────────────────
MODEL_DIR  = Path(__file__).parent / "blhd_model"
MAX_LENGTH = 64
PORT       = 5000
# ────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Cho phép HTML gọi API từ trình duyệt

# Load model 1 lần khi khởi động server
print(f"⏳ Đang load model từ: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.eval()
print("✅ Model sẵn sàng! Server đang chạy tại http://localhost:5000\n")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()

    if not text:
        return jsonify({"error": "Thiếu trường 'text'"}), 400

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0].tolist()
    label = 1 if probs[1] >= 0.5 else 0

    return jsonify({
        "text"    : text,
        "label"   : "BLHĐ" if label == 1 else "Bình thường",
        "is_blhd" : label == 1,
        "p_blhd"  : round(probs[1], 4),
        "p_normal": round(probs[0], 4),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "phobert-blhd"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
