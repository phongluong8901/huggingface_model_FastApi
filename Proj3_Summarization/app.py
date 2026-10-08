import re
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 1. Khởi tạo ứng dụng FastAPI với tiêu đề và mô tả
app = FastAPI(
    title="Text Summarization System", description="Summarize dialogue"
)

# 2. Tải mô hình T5 và Tokenizer đã được fine-tune từ thư mục cục bộ
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# 3. Đảm bảo mô hình được đặt lên thiết bị phù hợp (GPU nếu có, không thì dùng CPU)
device = "cuda" if model.device.type == "cuda" else "cpu"
model = model.to(device)

# 4. Thiết lập thư mục chứa các file giao diện HTML (Templates)
templates = Jinja2Templates(directory="templates")


# 5. Định nghĩa cấu trúc dữ liệu đầu vào (Schema) bằng Pydantic
class DialogueInput(BaseModel):
  dialogue: str


# 6. Hàm tiền xử lý và làm sạch văn bản thô
def clean_text(text):
  if not isinstance(text, str):
    return ""
  text = re.sub(
      r"\r\n", " ", text
  )  # Xóa ký tự xuống dòng đặc biệt của Windows
  text = re.sub(r"\s+", " ", text)  # Thay thế nhiều khoảng trắng liên tiếp bằng 1 khoảng trắng
  text = re.sub(r"<.*?>", "", text)  # Xóa các thẻ HTML (nếu có)
  text = text.strip().lower()  # Cắt bỏ khoảng trắng thừa 2 đầu và chuyển về chữ thường
  return text


# 7. Hàm xử lý tóm tắt hội thoại cốt lõi
def summarize_dialogue(dialogue: str) -> str:
  # Làm sạch nội dung đầu vào
  dialogue = clean_text(dialogue)

  # Mã hóa đoạn hội thoại thành các tensor số nguyên cho mô hình
  inputs = tokenizer(
      dialogue, return_tensors="pt", max_length=512, truncation=True
  )

  # Chuyển dữ liệu đầu vào sang đúng thiết bị (GPU/CPU) của mô hình
  inputs = {key: value.to(device) for key, value in inputs.items()}

  # Cho mô hình sinh ra chuỗi kết quả tóm tắt
  outputs = model.generate(
      inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True
  )

  # Giải mã các token số nguyên trả về thành văn bản ngôn ngữ tự nhiên
  summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return summary


# 8. API Endpoint nhận request POST từ giao diện web để tiến hành tóm tắt
@app.post("/summarize")
async def summarize(dialogue_input: DialogueInput):
  summary = summarize_dialogue(dialogue_input.dialogue)
  return {"summary": summary}


# 9. Giao diện trang chủ (HTTP GET) trả về file HTML cho người dùng
@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
  return templates.TemplateResponse(request, 'index.html')