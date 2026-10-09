import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Cấu hình giao diện trang Streamlit
st.set_page_config(
    page_title="Intent Recognition App",
    page_icon="🤖",
    layout="centered"
)

# Load model và tokenizer từ Hugging Face Hub (hoặc đổi thành đường dẫn thư mục local của bạn)
@st.cache_resource
def load_model_and_tokenizer():
    repo_id = "phong8901/distilbert-intent-recognition"  # Thay bằng repo của bạn nếu cần
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    return model, tokenizer

with st.spinner("Đang tải mô hình, vui lòng chờ trong giây lát..."):
    model, tokenizer = load_model_and_tokenizer()

# Định nghĩa nhãn ý định (Khớp chính xác với thứ tự lúc train)
id_to_label = {
    0: 'ratebook',
    1: 'playmusic',
    2: 'bookrestaurant',
    3: 'getweather',
    4: 'searchscreeningevent',
    5: 'addtoplaylist',
    6: 'searchcreativework'
}

# Giao diện Web App
st.title("🤖 Trợ Lý Nhận Diện Ý Định (Intent Recognition)")
st.write("Ứng dụng sử dụng mô hình **DistilBERT** đã được fine-tune để phân loại ý định câu lệnh của bạn.")

# Ô nhập liệu
text = st.text_input("Nhập câu lệnh của bạn:", placeholder="Ví dụ: Play my favorite upbeat songs...")

if st.button("Dự Đoán Ý Định", type="primary"):
    if text.strip():
        # Tokenize văn bản đầu vào
        inputs = tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=64, 
            return_tensors="pt"
        )

        # Suy luận (Inference)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Tính xác suất và lấy kết quả
        probs = F.softmax(logits, dim=-1)[0]
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        confidence = probs[predicted_class_id].item() * 100

        intent = id_to_label.get(predicted_class_id, "unknown intent")

        # Hiển thị kết quả ra màn hình
        st.success("Hoàn thành dự đoán!")
        st.markdown(f"### 🎯 Ý định: **{intent.upper()}**")
        st.info(f"📊 Độ tin cậy (Confidence): **{confidence:.2f}%**")
        
        # Hiển thị chi tiết xác suất các nhãn khác
        with st.expander("Xem chi tiết xác suất tất cả các ý định"):
            for idx, score in enumerate(probs):
                st.write(f"- `{id_to_label[idx]}`: {score.item() * 100:.2f}%")
                
    else:
        st.warning("Vui lòng nhập nội dung tin nhắn trước khi bấm dự đoán!")