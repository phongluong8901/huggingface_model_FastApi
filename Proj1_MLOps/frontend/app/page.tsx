"use client";

import Image from "next/image";
import { useEffect, useState } from "react";

// 1. Định nghĩa cấu trúc dữ liệu trả về cho từng mục lịch sử từ SQLite
interface HistoryItem {
  filename: string;
  predicted_label: string;
  confidence: number;
  timestamp: string;
}

export default function Home() {
  // 2. Các State quản lý trạng thái của ứng dụng (file, preview, loading, kết quả, lỗi, lịch sử)
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    predicted_idx: number;
    predicted_label: string;
    confidence: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  // 3. Hàm gọi API lấy danh sách lịch sử từ Backend FastAPI (kết nối SQLite)
  const fetchHistory = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/history/");
      if (res.ok) {
        const data = await res.json();
        setHistory(data.history);
      }
    } catch (err) {
      console.error("Không thể tải lịch sử:", err);
    }
  };

  // 4. Hook useEffect tự động nạp lịch sử ngay khi trang được mở lần đầu
  useEffect(() => {
    fetchHistory();
  }, []);

  // 5. Xử lý sự kiện khi người dùng chọn file ảnh từ máy tính
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file)); // Tạo URL ảo để hiển thị ảnh xem trước
      setResult(null);
      setError(null);
    }
  };

  // 6. Xử lý sự kiện gửi request (Submit) ảnh lên FastAPI Backend để dự đoán
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/infer/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Lỗi khi kết nối tới Server dự đoán!");
      }

      const data = await response.json();
      setResult(data);       // Lưu kết quả dự đoán trả về
      fetchHistory();        // Tải lại danh sách lịch sử ngay sau khi dự đoán xong
    } catch (err: any) {
      setError(err.message || "Có lỗi xảy ra");
    } finally {
      setLoading(false);
    }
  };

  return (
    // 7. Khung tổng thể giao diện cố định vừa khít màn hình (h-screen overflow-hidden)
    <main className="h-screen w-screen bg-gradient-to-br from-slate-50 via-emerald-50 to-teal-100 flex flex-col overflow-hidden p-4 lg:p-6">

      {/* Tiêu đề ứng dụng phía trên */}
      <header className="flex items-center justify-between px-6 py-3 bg-white/80 backdrop-blur-md rounded-2xl shadow-sm border border-white/40 mb-4 shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🌿</span>
          <h1 className="text-xl font-bold text-slate-800 tracking-tight">
            Hệ Thống Phân Loại Bệnh Lá Cây (ViT + ONNX)
          </h1>
        </div>
        <span className="text-xs font-semibold px-3 py-1 bg-emerald-100 text-emerald-800 rounded-full">
          AI Server Online
        </span>
      </header>

      {/* Bố cục chia 2 cột chính: Trái (Thao tác & Kết quả), Phải (Lịch sử SQLite) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 min-h-0">

        {/* CỘT TRÁI (Chiếm 7 phần): Giao diện chọn ảnh, xem trước và card kết quả */}
        <div className="lg:col-span-7 bg-white/90 backdrop-blur-md rounded-3xl shadow-xl border border-white/40 p-6 flex flex-col justify-between overflow-y-auto">
          <div>
            <h2 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
              📸 Tải lên ảnh lá cây chẩn đoán
            </h2>

            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Vùng chọn file và khung xem trước ảnh cạnh nhau */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
                <label className="flex flex-col items-center justify-center h-48 border-2 border-dashed border-slate-300 rounded-2xl cursor-pointer bg-slate-50 hover:bg-emerald-50/50 hover:border-emerald-400 transition-all p-4 text-center">
                  <svg className="w-8 h-8 mb-2 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <p className="text-xs font-medium text-slate-700">
                    <span className="text-emerald-600 font-semibold">Nhấn để chọn ảnh</span> hoặc kéo thả vào đây
                  </p>
                  <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
                </label>

                {/* Khung hiển thị ảnh xem trước */}
                <div className="relative h-48 w-full overflow-hidden rounded-2xl border border-slate-200 bg-slate-900 flex items-center justify-center">
                  {previewUrl ? (
                    <Image src={previewUrl} alt="Preview" fill className="object-contain" />
                  ) : (
                    <span className="text-xs text-slate-400">Chưa có ảnh được chọn</span>
                  )}
                </div>
              </div>

              {/* Nút bấm gửi dự đoán */}
              <button
                type="submit"
                disabled={!selectedFile || loading}
                className="w-full py-3 px-6 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 font-semibold text-white shadow-md hover:from-emerald-700 hover:to-teal-700 active:scale-[0.99] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "Đang phân tích mô hình AI..." : "🚀 Tiến hành dự đoán"}
              </button>
            </form>

            {/* Hiển thị lỗi nếu có */}
            {error && (
              <div className="mt-4 p-3 rounded-xl bg-red-50 border border-red-200 text-xs text-red-600 font-medium">
                {error}
              </div>
            )}
          </div>

          {/* Hiển thị kết quả dự đoán chi tiết kèm thanh tiến trình độ tin cậy */}
          {result && (
            <div className="mt-4 p-5 rounded-2xl bg-emerald-50 border border-emerald-200 shadow-sm">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-800">
                  Kết quả chẩn đoán gần nhất
                </span>
                <span className="px-2 py-0.5 text-xs font-bold text-emerald-900 bg-emerald-200 rounded-full">
                  Độ tin cậy: {result.confidence}%
                </span>
              </div>
              <p className="text-xl font-extrabold text-emerald-900">
                {result.predicted_label}
              </p>
              <div className="w-full bg-emerald-200/60 rounded-full h-2 mt-2 overflow-hidden">
                <div
                  className="bg-emerald-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${result.confidence}%` }}
                ></div>
              </div>
            </div>
          )}
        </div>

        {/* CỘT PHẢI (Chiếm 5 phần): Khu vực hiển thị danh sách lịch sử có thanh cuộn riêng */}
        <div className="lg:col-span-5 bg-white/90 backdrop-blur-md rounded-3xl shadow-xl border border-white/40 p-6 flex flex-col min-h-0">
          <h2 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2 shrink-0">
            📜 Lịch sử chẩn đoán (SQLite)
          </h2>

          <div className="flex-1 overflow-y-auto space-y-3 pr-1">
            {history.length === 0 ? (
              <div className="h-full flex items-center justify-center text-slate-400 text-sm">
                Chưa có lịch sử dự đoán nào.
              </div>
            ) : (
              history.map((item, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-slate-50 hover:bg-slate-100 rounded-xl border border-slate-200 text-xs transition-all"
                >
                  <div className="space-y-1 truncate mr-2">
                    <p className="font-semibold text-slate-700 truncate max-w-[140px] sm:max-w-[200px]">
                      {item.filename}
                    </p>
                    <span className="text-[10px] text-slate-400 block">
                      {item.timestamp}
                    </span>
                  </div>
                  <div className="text-right shrink-0">
                    <span className="font-bold text-emerald-700 block">
                      {item.predicted_label}
                    </span>
                    <span className="text-[11px] font-semibold text-slate-600">
                      {item.confidence}%
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

      </div>
    </main>
  );
}