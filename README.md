# ⚖️ Chatbot Tra cứu Luật An toàn, vệ sinh lao động 2015

Dự án môn Trí tuệ nhân tạo: xây dựng hệ thống tra cứu tự động **Luật An toàn, vệ sinh lao động 2015 (Luật số: 84/2015/QH13)** bằng kỹ thuật tiền xử lý văn bản, trích xuất từ khóa (TF-IDF) và tìm kiếm ngữ nghĩa (Sentence Embedding).

## 📖 Giới thiệu

Hệ thống đọc trực tiếp file luật gốc (`.docx`), tự động tách văn bản thành các đơn vị Điều / Khoản / Điểm, rồi cho phép người dùng đặt câu hỏi bằng tiếng Việt tự nhiên và nhận lại đúng điều khoản liên quan — thông qua giao diện web Streamlit.

Hệ thống trả lời theo **3 tầng xử lý**, ưu tiên từ trên xuống:

1. **Tầng 0 — Metadata:** trả lời các câu hỏi về văn bản (cơ quan ban hành, ngày ban hành, số hiệu, hiệu lực...).
2. **Tầng 1 — Tra cứu trực tiếp:** khi người dùng hỏi thẳng "Điều X", "Khoản Y", "Điểm z" → dùng regex để lấy đúng nguyên văn.
3. **Tầng 2 — Tìm kiếm ngữ nghĩa (semantic search):** khi câu hỏi diễn đạt tự do → nhúng câu hỏi bằng mô hình embedding, so khớp cosine similarity với toàn bộ 606 đoạn luật đã được chunk sẵn.

> ⚠️ **Lưu ý về phạm vi hiện tại:** hệ thống hiện chỉ trả về **nguyên văn điều luật** tìm được (retrieval thuần), **chưa** tích hợp LLM (Gemini/GPT...) để diễn giải lại câu trả lời bằng ngôn ngữ tự nhiên. Đây là hướng phát triển tiếp theo.

## 🚀 Tính năng chính

- **Tiền xử lý văn bản pháp lý** (`src/preprocess.py`): đọc file `.docx` gốc bằng `docx2txt`, chuẩn hoá lỗi font/khoảng trắng bằng regex, tách thành **606 chunks** theo cấu trúc Chương → Điều → Khoản → Điểm.
- **Trích xuất từ khóa pháp lý** (`src/keyphrase.py`): dùng `TF-IDF` kết hợp N-gram (1–18) trên corpus đã tách từ bằng `underthesea`, sinh ra bộ 150 keyphrase pháp lý (`keyphrase_atvsld.csv`).
- **Tìm kiếm ngữ nghĩa** (`src/retrieval.py`): nhúng văn bản bằng mô hình `intfloat/multilingual-e5-large` (đổi được sang bản `-base` nếu máy yếu qua biến môi trường `MODEL_E5`), có từ điển đồng nghĩa mở rộng câu hỏi (vd. `tnlđ` → `tai nạn lao động`) và cache embedding ra `.npy` để chạy lại nhanh.
- **Đánh giá định lượng** (`src/evaluation.py`): chấm điểm hệ thống trên bộ **60 câu hỏi kiểm thử** (`data/test_questions.csv`) bằng chỉ số **Hit@1 / Hit@3 / Hit@5 / MRR**, so sánh với baseline TF-IDF, tự động xuất biểu đồ (PNG) và bảng (CSV) vào `reports/hinh-anh/`.
- **Giao diện người dùng** (`app/app.py`): web app hỏi-đáp tương tác bằng `Streamlit`.

### Kết quả đánh giá (60 câu hỏi test, top-5)

| Phương pháp | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| TF-IDF (baseline) | 0.567 | 0.733 | 0.750 | 0.643 |
| **E5 embedding** | **0.733** | **0.850** | **0.917** | **0.807** |

*(số liệu lấy từ `reports/hinh-anh/bang-so-sanh-phuong-phap.csv`, có thể chạy lại bằng `python src/evaluation.py`)*

## 📂 Cấu trúc thư mục thực tế

```text
AI_law_retrieval/
│
├── app/
│   └── app.py                        # Giao diện Streamlit (đọc dữ liệu + embedding + tra cứu 3 tầng)
│
├── data/
│   ├── raw/
│   │   └── 84_2015_QH13.docx         # File luật gốc
│   ├── processed/
│   │   ├── law_dataset_chunks.csv    # 606 chunks sau khi tách Điều/Khoản/Điểm
│   │   └── keyphrase_atvsld.csv      # 150 keyphrase pháp lý đã trích xuất
│   └── test_questions.csv            # 60 câu hỏi kiểm thử (có đáp án đúng để chấm điểm)
│
├── model/                            # Nơi lưu cache embedding (models/e5_embeddings.npy) — sinh ra khi chạy, không commit
│
├── notebooks/
│   └── law.ipynb                     # Notebook thực nghiệm gốc (chạy trên Colab)
│
├── reports/
│   └── hinh-anh/                     # Biểu đồ (PNG) + bảng kết quả (CSV) do evaluation.py sinh ra
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py                 # .docx -> chuẩn hoá -> 606 chunks
│   ├── keyphrase.py                  # TF-IDF + N-gram (1-18) -> keyphrase
│   ├── retrieval.py                  # Lõi tra cứu 3 tầng (dùng lại được cho cả CLI lẫn app)
│   └── evaluation.py                 # Chấm Hit@1/3/5 + MRR, xuất biểu đồ/bảng
│
├── requirements.txt
└── README.md
```

## ⚙️ Cài đặt

```bash
git clone https://github.com/vinhdo19111999-hash/AI_law_retrieval.git
cd AI_law_retrieval
pip install -r requirements.txt
```

## ▶️ Cách chạy

**1. Tạo lại dữ liệu chunks từ file luật gốc (nếu chưa có sẵn trong `data/processed/`):**
```bash
python src/preprocess.py
python src/keyphrase.py
```

**2. Chạy thử tra cứu bằng dòng lệnh:**
```bash
# Nhanh, không cần tải model embedding (chỉ tầng 0 + tầng 1)
python src/retrieval.py --khong-embedding "Điều 6 quy định gì?"

# Đầy đủ 3 tầng (sẽ tải model intfloat/multilingual-e5-large)
python src/retrieval.py "Người lao động bị tai nạn lao động suy giảm 81% được bồi thường bao nhiêu?"
```

**3. Chạy giao diện web:**
```bash
streamlit run app/app.py
```
Máy yếu, RAM thấp → đặt biến môi trường trước khi chạy để dùng bản model nhẹ hơn:
```bash
export MODEL_E5=intfloat/multilingual-e5-base
```

**4. Chạy đánh giá định lượng:**
```bash
python src/evaluation.py
```
Kết quả (Hit@1/3/5, MRR, biểu đồ so sánh TF-IDF vs E5) sẽ được ghi vào `reports/hinh-anh/`.

## 🚧 Hướng phát triển

- Tích hợp một LLM (ví dụ Gemini API) ở tầng cuối để **diễn giải lại** đoạn luật tìm được thành câu trả lời tự nhiên, thay vì chỉ trả nguyên văn.
- Mở rộng bộ đánh giá để chấm luôn tầng 0 (metadata) và tầng 1 (tra cứu trực tiếp Điều/Khoản).
- Thử nghiệm và chọn ngưỡng cosine similarity (`nguong`) dựa trên thực nghiệm quét nhiều giá trị, thay vì đặt cố định 0.25.

## 📄 Giấy phép / Nguồn dữ liệu

Văn bản luật sử dụng trong dự án là văn bản pháp luật công khai: **Luật An toàn, vệ sinh lao động số 84/2015/QH13**, ban hành bởi Quốc hội ngày 25/06/2015.
