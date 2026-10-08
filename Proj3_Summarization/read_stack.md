# --- lib
1. transformers
Loại: Thư viện mã nguồn mở của Hugging Face về Học máy / Xử lý ngôn ngữ tự nhiên (NLP).

Tác dụng: Đây là thư viện cốt lõi cung cấp toàn bộ các kiến trúc mô hình học sâu tiên tiến (như T5, BERT, GPT, v.v.). Nó cho phép bạn dễ dàng tải các mô hình pre-trained, fine-tune chúng trên dữ liệu tùy chỉnh, và gọi các pipeline sinh văn bản (model.generate()) chỉ với vài dòng code.

2. sentencepiece
Loại: Thư viện mã hóa phụ từ (Subword Tokenization) của Google.

Tác dụng: Các mô hình như T5 sử dụng thuật toán của sentencepiece để tách chữ thành các token. Thư viện này là bắt buộc phải cài đặt nếu bạn dùng họ nhà mô hình T5, vì nó giúp bộ Tokenizer hiểu được các quy tắc cắt chữ, nối từ và quản lý từ điển của mô hình.

3. jinja2
Loại: Thư viện lập bản mẫu (Template Engine) cho Python.

Tác dụng: Cho phép bạn kết hợp dữ liệu từ Python (backend) vào bên trong các file giao diện HTML (frontend) một cách linh hoạt, giúp FastAPI dễ dàng render ra các trang web động khi người dùng truy cập vào đường dẫn chính (/).

# --- import
rom fastapi import FastAPI, Request: Khởi tạo ứng dụng FastAPI và xử lý đối tượng Request trong các HTTP request.

from fastapi.responses import HTMLResponse: Dùng để trả về các phản hồi dạng mã HTML trực tiếp cho trình duyệt.

from fastapi.templating import Jinja2Templates: Hỗ trợ liên kết FastAPI với các file template HTML (như file index.html đã làm ở bước trước).

from pydantic import BaseModel: Dùng để tạo các lớp dữ liệu đầu vào (data schema). Giúp FastAPI tự động kiểm tra định dạng JSON mà người dùng gửi lên có đúng cấu trúc hay không (ví dụ: bắt buộc phải có trường dialogue kiểu chuỗi chữ).

from transformers import T5Tokenizer, T5ForConditionalGeneration: Tải bộ mã hóa từ (T5Tokenizer) và mô hình sinh văn bản (T5ForConditionalGeneration) dòng T5 đã được fine-tune để tiến hành xử lý dự đoán.

import re: Thư viện chuẩn của Python về Biểu thức chính quy (Regular Expression). Thư viện này rất hữu ích nếu bạn cần dùng hàm clean_text để lọc bỏ các ký tự rác, các thẻ HTML hoặc các dấu cách thừa trong đoạn hội thoại trước khi đưa vào mô hình.