"""Mã nguồn lõi của dự án Chatbot tra cứu Luật An toàn, vệ sinh lao động 2015.

Các module:
    preprocess  — đọc .docx -> chuẩn hoá -> tách thành 606 chunks
    keyphrase   — trích keyphrase pháp lý bằng TF-IDF + n-gram (1→18)
    retrieval   — lõi tra cứu 3 tầng (metadata / trực tiếp / ngữ nghĩa E5)
    evaluation  — chấm Hit@1/3/5 + MRR trên bộ 60 câu hỏi

Chạy nhanh từ thư mục gốc repo:
    python src/preprocess.py && python src/keyphrase.py
    python src/retrieval.py --khong-embedding "Điều 6 quy định gì?"
    python src/evaluation.py
"""
