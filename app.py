import os
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS  # <-- 1. IMPORT THÊM CORS
from pinecone import Pinecone

# --- CẤU HÌNH VÀ KẾT NỐI PINECONE ---
# (Giữ nguyên không đổi)
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_HOST = os.environ.get('PINECONE_HOST')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')

if not all([PINECONE_API_KEY, PINECONE_HOST, PINECONE_INDEX_NAME]):
    raise ValueError("Lỗi: Vui lòng thiết lập các biến môi trường PINECONE_API_KEY, PINECONE_HOST, và PINECONE_INDEX_NAME")

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(name=PINECONE_INDEX_NAME, host=PINECONE_HOST)
    print(f"Kết nối thành công tới index '{PINECONE_INDEX_NAME}'")
except Exception as e:
    raise RuntimeError(f"Không thể khởi tạo kết nối Pinecone: {e}")

# --- TẠO ỨNG DỤNG WEB FLASK ---
app = Flask(__name__)

# --- KÍCH HOẠT CORS ---
CORS(app)  # <-- 2. KÍCH HOẠT CORS CHO TOÀN BỘ ỨNG DỤNG
# Dòng này sẽ cho phép tất cả các tên miền khác gọi tới API của bạn.

# --- ĐỊNH NGHĨA CÁC ROUTE ---
# (Giữ nguyên không đổi)
@app.route('/')
def health_check():
    try:
        stats = index.describe_index_stats()
        return jsonify({
            "message": "Ứng dụng đang hoạt động và đã kết nối thành công tới Pinecone.",
            "index_name": PINECONE_INDEX_NAME,
            "index_stats": stats.to_dict()
        })
    except Exception as e:
        return jsonify({"error": f"Không thể lấy thông tin từ Pinecone: {e}"}), 500

@app.route('/query')
def query_example():
    try:
        random_vector = np.random.rand(768).tolist()
        query_results = index.query(
            vector=random_vector,
            top_k=5,
            include_metadata=True
        )
        return jsonify(query_results.to_dict())
    except Exception as e:
        return jsonify({"error": f"Lỗi xảy ra trong quá trình truy vấn: {e}"}), 500

# --- CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

