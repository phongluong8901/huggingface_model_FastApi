from pydantic import BaseModel, Field

# Định nghĩa khung dữ liệu đầu vào (Request Body) từ client gửi lên
class SentimentRequest(BaseModel):
    text: str = Field(..., description="Text for sentiment analysis")

# Định nghĩa khung dữ liệu đầu ra (Response Body) trả về cho client
class SentimentResponse(BaseModel):
    sentiment: str # Kết quả cảm xúc (ví dụ: Positive, Negative, Neutral) dạng chuỗi
    confidence: float # Độ tin cậy của mô hình dưới dạng số thực (ví dụ: 0.9856)
    model: str # Tên mô hình được sử dụng để phân tích