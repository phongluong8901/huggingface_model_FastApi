from datetime import datetime
import sqlite3

DB_PATH = "./history.db"


# Khởi tạo bảng lịch sử dự đoán nếu chưa tồn tại
def init_db():
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            predicted_label TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
  conn.commit()
  conn.close()


# Hàm lưu kết quả dự đoán vào SQLite
def save_prediction(filename: str, predicted_label: str, confidence: float):
  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  cursor.execute(
      """
        INSERT INTO predictions (filename, predicted_label, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    """,
      (filename, predicted_label, confidence, timestamp),
  )
  conn.commit()
  conn.close()


# Hàm lấy danh sách lịch sử dự đoán
def get_prediction_history(limit: int = 20):
  conn = sqlite3.connect(DB_PATH)
  conn.row_factory = sqlite3.Row  # Trả về dạng từ điển để dễ chuyển thành JSON
  cursor = conn.cursor()
  cursor.execute(
      "SELECT filename, predicted_label, confidence, timestamp FROM predictions"
      f" ORDER BY id DESC LIMIT {limit}"
  )
  rows = cursor.fetchall()
  conn.close()
  return [dict(row) for row in rows]