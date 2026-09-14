# ⚖️ Chatbot Tra cứu Luật An toàn, vệ sinh lao động 2015

## 📖 Giới thiệu
Đây là dự án ứng dụng Xử lý ngôn ngữ tự nhiên (NLP) và Mô hình ngôn ngữ lớn (LLM) để xây dựng hệ thống hỏi đáp, tra cứu thông tin pháp lý tự động. Cụ thể, dự án tập trung vào **Luật An toàn, vệ sinh lao động 2015 (Luật số: 84/2015/QH13)**. 

Hệ thống giúp tự động hóa quá trình trích xuất và tìm kiếm thông tin, hỗ trợ đắc lực cho công tác quản lý EHS/HSE tại các doanh nghiệp trong việc nắm bắt, tuân thủ và áp dụng chính xác các quy định về an toàn lao động.

## 🚀 Các tính năng chính (Features)
- **Tiền xử lý văn bản pháp lý:** Đọc và phân chia cấu trúc file luật (Quyền, Nghĩa vụ, Điều kiện,...) thành các đoạn nhỏ (chunks) tối ưu cho tìm kiếm.
- **Trích xuất từ khóa pháp lý (Keyphrase Extraction):** Sử dụng thuật toán `TF-IDF` kết hợp `N-gram (1-18)` để nhận diện các cụm từ quan trọng mang tính chuyên ngành.
- **Tìm kiếm ngữ nghĩa (Semantic Search):** Nhúng văn bản (Embedding) bằng mô hình `intfloat/multilingual-e5-large` để tìm ra chính xác điều luật liên quan đến câu hỏi của người dùng.
- **Tích hợp LLM sinh câu trả lời:** Sử dụng API của Gemini để tổng hợp ngữ cảnh pháp lý thu được và sinh ra câu trả lời tự nhiên, dễ hiểu, bám sát luật.
- **Giao diện người dùng trực quan:** Xây dựng web app tương tác trực tiếp bằng `Streamlit`.

## 📂 Cấu trúc thư mục (Project Structure)
```text
NLP_law_retrieval/
│
├── data/
│   ├── 84_2015_QH13_281961.docx      # File luật gốc
│   ├── law_dataset_chunks.csv        # Dữ liệu luật sau khi chunking
│   └── keyphrase_atvsld.csv          # Dữ liệu từ khóa đã trích xuất
│
├── notebooks/
│   └── law_retrieval_colab.ipynb     # File thực nghiệm thuật toán trên Colab
│
├── src/
│   └── app.py                        # Mã nguồn chính của giao diện Streamlit chatbot
│
├── .env.example                      # File mẫu chứa biến môi trường (API Keys)
├── .gitignore                        # Cấu hình bỏ qua các file không cần push (như .env)
├── requirements.txt                  # Danh sách các thư viện cần thiết
└── README.md                         # Tài liệu hướng dẫn (File này)
