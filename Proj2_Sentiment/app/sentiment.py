# app/sentiment.py
from transformers import pipeline  # Nhập hàm pipeline từ thư viện Hugging Face Transformers để dễ dàng tải và sử dụng mô hình AI
import threading                  # Nhập thư viện threading để xử lý đồng bộ hóa đa luồng (multi-threading)

_MODEL = None                     # Biến toàn cục (private) dùng để lưu trữ model AI sau khi được tải lần đầu tiên
_LOCK = threading.Lock()          # Khóa (Lock) an toàn luồng, đảm bảo chỉ có 1 tiến trình/luồng tải model trong một thời điểm

def get_sentiment_model():        # Hàm thiết kế theo mẫu Singleton (Lazy Loading) để tải model một lần duy nhất dùng chung
    global _MODEL                 # Khai báo sử dụng biến toàn cục _MODEL

    if _MODEL is None:            # Kiểm tra nhanh lần 1 (không mất phí lock) xem model đã được tải chưa
        with _LOCK:               # Bắt đầu khóa an toàn để tránh việc nhiều luồng cùng tải model cùng lúc (Double-checked locking)
            if _MODEL is None:    # Kiểm tra lại lần 2 bên trong khóa chắc chắn model vẫn chưa có
                # Tải pipeline phân tích cảm xúc, chỉ định model Hugging Face và ép chạy trên CPU (device=-1)
                _MODEL = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

    return _MODEL                 # Trả về đối tượng model AI đã được khởi tạo

def analyze_sentiment(text: str): # Hàm nhận vào một chuỗi văn bản và thực hiện phân tích cảm xúc
    model = get_sentiment_model() # Gọi hàm lấy model (nếu chưa tải thì sẽ tải, nếu tải rồi sẽ dùng luôn)
    result = model(text)[0]       # Đưa văn bản qua model AI để dự đoán, lấy phần tử đầu tiên kết quả trả về ([0])

    label = result["label"]       # Lấy nhãn cảm xúc gốc từ model (ví dụ: "POSITIVE" hoặc "NEGATIVE")
    score = float(result["score"]) # Lấy độ tin cậy (xác suất) của dự đoán và ép kiểu sang float

    if score < 0.6:               # Nếu độ tin cậy thấp hơn 0.6 (không chắc chắn)
        sentiment = "Neutral"     # Gán nhãn là trung tính (Neutral)
    elif label == "POSITIVE":     # Nếu nhãn gốc là POSITIVE và độ tin cậy đủ cao
        sentiment = "Positive"    # Gán nhãn tích cực (Positive)
    else:                         # Ngược lại (nhãn là NEGATIVE và độ tin cậy cao)
        sentiment = "Negative"    # Gán nhãn tiêu cực (Negative)

    return {                      # Trả về kết quả dưới dạng từ điển (dictionary)
        "sentiment": sentiment,   # Chuỗi cảm xúc đã được chuẩn hóa ("Positive", "Negative", hoặc "Neutral")
        "confidence": score       # Điểm độ tin cậy của mô hình
    }