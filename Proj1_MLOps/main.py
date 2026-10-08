from contextlib import asynccontextmanager
import io
# Nhập các hàm thao tác SQLite từ file db_sqlite của bạn
from db_sqlite import get_prediction_history, init_db, save_prediction
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as rt
from PIL import Image

# 1. Đường dẫn tới file mô hình ONNX và khai báo biến toàn cục chứa session ONNX Runtime
ONNX_MODEL_PATH = "./vit_classification.onnx"
ort_session = None

# 2. Định nghĩa từ điển nhãn (Labels) ánh xạ từ chỉ số model sang tên bệnh thực tế
LABELS = {
    0: "Healthy (Khỏe mạnh)",
    1: "Powdery (Bệnh phấn trắng)",
    2: "Rust (Bệnh gỉ sắt)",
}


# 3. Quản lý vòng đời ứng dụng: Khởi tạo DB và nạp mô hình ONNX một lần duy nhất khi server bật
@asynccontextmanager
async def lifespan(app: FastAPI):
  global ort_session
  init_db()  # Tạo bảng SQLite nếu chưa tồn tại
  ort_session = rt.InferenceSession(ONNX_MODEL_PATH)  # Nạp mô hình vào RAM
  yield
  ort_session = None  # Giải phóng tài nguyên khi tắt server


# 4. Khởi tạo ứng dụng FastAPI kết nối với vòng đời lifespan ở trên
app = FastAPI(lifespan=lifespan)

# 5. Cấu hình CORS để cho phép Frontend (Next.js) gọi API khác cổng mà không bị trình duyệt chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 6. Hàm tiền xử lý ảnh: Đọc, resize về 224x224, chuyển định dạng kênh màu CHW và chuẩn hóa về [0.0, 1.0]
def preprocess_image(image_bytes):
  img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
  resized_img = img.resize((224, 224))
  np_array = np.asarray(resized_img)
  img_transposed = np.transpose(np_array, (2, 0, 1))
  img_expanded = np.expand_dims(img_transposed, axis=0)
  img_normalized = img_expanded.astype(np.float32) / 255.0
  return img_normalized


# 7. Endpoint nhận request POST file ảnh để tiến hành suy luận (inference) mô hình AI
@app.post("/infer/")
async def infer(file: UploadFile = File(...)):
  if ort_session is None:
    return {"error": "Model is not loaded"}

  image_bytes = await file.read()
  image_preprocessed = preprocess_image(image_bytes)

  # Đóng gói input và thực thi suy luận bằng ONNX Runtime
  session_input = {ort_session.get_inputs()[0].name: image_preprocessed}
  onnx_output = ort_session.run(None, session_input)
  onnx_logits = onnx_output[0][0]

  # Dùng hàm Softmax để chuyển đổi logits thô thành xác suất phần trăm độ tin cậy
  exp_logits = np.exp(onnx_logits - np.max(onnx_logits))
  probabilities = exp_logits / np.sum(exp_logits)

  # Lấy index lớn nhất, ánh xạ tên nhãn và tính phần trăm độ tin cậy
  pred_idx = int(np.argmax(probabilities))
  predicted_label = LABELS.get(pred_idx, "Unknown")
  confidence = round(float(probabilities[pred_idx]) * 100, 2)

  # Lưu kết quả dự đoán vào cơ sở dữ liệu SQLite thông qua module db_sqlite
  save_prediction(file.filename, predicted_label, confidence)

  return {
      "predicted_idx": pred_idx,
      "predicted_label": predicted_label,
      "confidence": confidence,
  }


# 8. Endpoint GET lấy danh sách lịch sử chẩn đoán từ SQLite (mặc định lấy 20 bản ghi mới nhất)
@app.get("/history/")
def get_history():
  history_list = get_prediction_history(limit=20)
  return {"history": history_list}


# 9. Endpoint kiểm tra trạng thái hoạt động của server
@app.get("/")
def test_api():
  return {"message": "API is working"}