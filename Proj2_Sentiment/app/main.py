from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.models import SentimentResponse, SentimentRequest
from app.sentiment import analyze_sentiment
import os

app = FastAPI(title="Sentiment Analysis API")

# Lấy đường dẫn tuyệt đối đến thư mục static nằm ngoài app/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount thư mục static để phục vụ các tệp giao diện
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def home():
    # Trả về file index.html khi truy cập trang chủ
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# Khai báo endpoint POST tại "/analyze", định nghĩa kiểu dữ liệu trả về chuẩn
@app.post("/analyze", response_model=SentimentResponse)
async def analyze(payload: SentimentRequest):   # Hàm nhận dữ liệu từ client dưới dạng đối tượng SentimentRequest
    if not payload.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    # Gọi hàm AI để thực hiện phân tích cảm xúc dựa trên đoạn text nhận được
    result = analyze_sentiment(payload.text)

    # Trả về kết quả sau khi đã định dạng lại (làm tròn số và đóng gói vào đối tượng SentimentResponse)
    return SentimentResponse(
        sentiment=result["sentiment"],
        confidence=round(result["confidence"], 4),
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )