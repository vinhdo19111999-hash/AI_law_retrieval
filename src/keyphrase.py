"""
src/keyphrase.py — Trích xuất keyphrase pháp lý bằng TF-IDF + n-gram (1→18).
Ghi ra:
    data/processed/law_dataset_chunks.csv   (ghi đè, thêm 2 cột -> 16 cột)
    data/processed/keyphrase_atvsld.csv     (150 dòng × 4 cột)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from underthesea import word_tokenize

GOC_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUONG_DAN_CHUNKS = os.path.join(GOC_REPO, "data", "processed", "law_dataset_chunks.csv")
DUONG_DAN_KEYPHRASE = os.path.join(GOC_REPO, "data", "processed", "keyphrase_atvsld.csv")


def _chuan_hoa_kieu_cot_truoc_khi_ghi(df: pd.DataFrame) -> pd.DataFrame:
    """Giữ cột định danh ở dạng SỐ NGUYÊN khi ghi CSV (tránh bị thành "4.0").

    Vì sao cần: cột khoan_id có ô trống (các dòng dieu_intro) nên khi pandas đọc lại CSV
    sẽ thành float -> ghi ra "4.0" thay vì "4". Ép sang Int64 (kiểu số nguyên cho phép trống)
    giúp file sinh ra GIỐNG file do notebook tạo, và chạy lại nhiều lần vẫn ổn định.
    """
    for cot in ("dieu_id", "khoan_id"):
        if cot in df.columns:
            try:
                df[cot] = df[cot].astype("Int64")
            except (TypeError, ValueError):
                pass          # cột có ký tự không phải số -> giữ nguyên
    return df


def lam_giau_keyphrase(df_chunks: pd.DataFrame,
                       duong_dan_chunks: str | None = None,
                       duong_dan_keyphrase: str | None = None,
                       luu: bool = True):
    """Trích keyphrase + gán nhãn dạng quy định, thêm 2 cột vào df_chunks.

    Trả về (df_chunks, df_keyphrase_loc, KEYPHRASE_ATVSLD_AUTO).
    """
    duong_dan_chunks = duong_dan_chunks or DUONG_DAN_CHUNKS
    duong_dan_keyphrase = duong_dan_keyphrase or DUONG_DAN_KEYPHRASE
    # ============================================================
    # PHẦN 1: TRÍCH XUẤT KEYPHRASE TỰ ĐỘNG TỪ TOÀN BỘ CORPUS
    # Sử dụng TF-IDF với n-gram từ 1 đến 18 để bao phủ cả thuật
    # ngữ pháp lý rất dài như "máy thiết bị vật tư chất có yêu cầu
    # nghiêm ngặt về an toàn vệ sinh lao động" (13 từ sau tách từ)
    # ============================================================

    # --- 1.1 Hư từ tiếng Việt (loại bỏ trước khi trích xuất) ---
    HU_TU_KEYPHRASE = """và của cho theo là các những này đó khi nếu thì được không
    phải để trong với từ đến bị do vì như sau trước mà nào nên hay hoặc tại sẽ đã về
    một hai ba bốn năm sáu bảy tám chín mười việc quy định tại khoản điều chi tiết
    người điểm mục chương luật số năm tháng ngày hoặc theo còn lại trên dưới""".split()  # hư từ tiếng Việt cần loại bỏ khỏi keyphrase

    def tach_tu_cho_keyphrase(van_ban: str) -> str:                                    # hàm tách từ dùng riêng cho trích keyphrase
        """Tách từ tiếng Việt bằng underthesea, chuyển thường.
        VD: 'Tai nạn lao động' -> 'tai_nạn lao_động'
        Gạch dưới giúp underthesea nhận diện từ ghép chính xác."""
        return word_tokenize(str(van_ban).lower(), format="text")                      # tách từ + viết thường + gạch dưới từ ghép

    # --- 1.2 Chuẩn bị corpus đã tách từ ---
    print("=" * 80)                                                                    # đường kẻ phân cách
    print("BƯỚC 1: TÁCH TỪ TOÀN BỘ CORPUS ĐỂ TRÍCH XUẤT KEYPHRASE")                  # tiêu đề bước 1
    print("=" * 80)                                                                    # đường kẻ phân cách
    corpus_cho_keyphrase = df_chunks["noi_dung"].astype(str).tolist()                  # lấy cột nội dung thành danh sách chuỗi
    corpus_tach_tu_kp = [tach_tu_cho_keyphrase(v) for v in corpus_cho_keyphrase]       # tách từ từng dòng nội dung
    print(f"Đã tách từ {len(corpus_tach_tu_kp)} dòng nội dung.")                      # in số dòng đã xử lý

    # --- 1.3 Trích xuất keyphrase bằng TF-IDF n-gram (1→18) ---
    print("\n" + "=" * 80)                                                              # đường kẻ phân cách
    print("BƯỚC 2: TÍNH TF-IDF VỚI N-GRAM TỪ 1 ĐẾN 18")                               # tiêu đề bước 2
    print("=" * 80)                                                                     # đường kẻ phân cách
    print("Giải thích tham số:")                                                         # in giải thích
    print("  ngram_range=(1, 18): tạo cụm từ từ 1 từ đến 18 từ liên tiếp")              # giải thích ngram_range
    print("    -> Vì thuật ngữ ATVSLĐ có thể dài đến 13-14 từ sau tách từ")             # lý do chọn 18
    print("    -> VD: 'máy thiết_bị vật_tư chất yêu_cầu nghiêm_ngặt an_toàn")          # ví dụ cụm dài
    print("           vệ_sinh lao_động' = 10 token sau tách từ underthesea")             # giải thích thêm
    print("  max_df=0.85: bỏ cụm xuất hiện trong >85% tài liệu (quá chung)")            # giải thích max_df
    print("  min_df=2: chỉ giữ cụm xuất hiện ít nhất 2 lần (loại nhiễu)")               # giải thích min_df
    print("  max_features=8000: giới hạn tối đa 8000 cụm từ (tiết kiệm RAM)")           # giải thích max_features
    print("  sublinear_tf=True: dùng log(1+TF) thay vì TF thô")                         # giải thích sublinear_tf
    print("    -> Giảm ảnh hưởng của cụm từ lặp quá nhiều trong 1 chunk")                # giải thích tác dụng

    tfidf_kp = TfidfVectorizer(                                                          # khởi tạo bộ TF-IDF cho keyphrase
        ngram_range=(1, 18),                                                               # n-gram từ unigram đến 18-gram
        stop_words=HU_TU_KEYPHRASE,                                                        # loại bỏ hư từ tiếng Việt
        token_pattern=r"(?u)\b\w[\w_]*\b",                                                 # nhận cả từ có gạch dưới (từ ghép underthesea)
        max_df=0.85,                                                                        # bỏ cụm quá phổ biến (>85% tài liệu)
        min_df=2,                                                                            # chỉ giữ cụm xuất hiện ≥2 lần
        max_features=8000,                                                                   # giới hạn tối đa 8000 cụm từ
        sublinear_tf=True,                                                                   # dùng log(1+TF) để giảm bias tần suất cao
    )
    X_kp = tfidf_kp.fit_transform(corpus_tach_tu_kp)                                       # học từ vựng + vector hoá toàn bộ corpus
    print(f"\nĐã tạo ma trận TF-IDF: {X_kp.shape[0]} tài liệu × {X_kp.shape[1]} cụm từ") # in kích thước ma trận

    # --- 1.4 Đếm tần suất xuất hiện thô (bổ sung cùng tham số) ---
    print("\n" + "=" * 80)                                                                  # đường kẻ phân cách
    print("BƯỚC 3: ĐẾM TẦN SUẤT XUẤT HIỆN THÔ CỦA TỪNG N-GRAM")                           # tiêu đề bước 3
    print("=" * 80)                                                                         # đường kẻ phân cách

    count_kp = CountVectorizer(                                                              # khởi tạo bộ đếm tần suất
        ngram_range=(1, 18),                                                                  # cùng phạm vi n-gram với TF-IDF
        stop_words=HU_TU_KEYPHRASE,                                                           # cùng danh sách hư từ
        token_pattern=r"(?u)\b\w[\w_]*\b",                                                    # cùng mẫu token
        max_df=0.85,                                                                           # cùng ngưỡng max_df
        min_df=2,                                                                               # cùng ngưỡng min_df
        max_features=8000,                                                                      # cùng giới hạn max_features
    )
    X_count = count_kp.fit_transform(corpus_tach_tu_kp)                                       # học từ vựng + đếm tần suất
    print(f"Đã đếm tần suất: {X_count.shape[1]} cụm từ")                                     # in số cụm từ đã đếm

    # --- 1.5 Tổng hợp điểm TF-IDF + tần suất cho toàn bộ corpus ---
    print("\n" + "=" * 80)                                                                     # đường kẻ phân cách
    print("BƯỚC 4: TỔNG HỢP VÀ XẾP HẠNG KEYPHRASE")                                          # tiêu đề bước 4
    print("=" * 80)                                                                            # đường kẻ phân cách

    ten_cum_tu_tfidf = tfidf_kp.get_feature_names_out()                                        # lấy tên cụm từ từ bộ TF-IDF
    diem_tfidf_tong = np.asarray(X_kp.sum(axis=0)).flatten()                                   # tổng điểm TF-IDF mỗi cụm trên toàn corpus

    ten_cum_tu_count = count_kp.get_feature_names_out()                                        # lấy tên cụm từ từ bộ đếm
    tan_suat_tong = np.asarray(X_count.sum(axis=0)).flatten()                                  # tổng tần suất mỗi cụm trên toàn corpus

    df_tfidf_score = pd.DataFrame({                                                            # tạo bảng điểm TF-IDF
        "keyphrase": ten_cum_tu_tfidf,                                                          # cột tên cụm từ
        "tfidf_score": diem_tfidf_tong,                                                         # cột tổng điểm TF-IDF
    }).sort_values("tfidf_score", ascending=False)                                               # sắp xếp giảm dần

    df_count_score = pd.DataFrame({                                                              # tạo bảng tần suất
        "keyphrase": ten_cum_tu_count,                                                            # cột tên cụm từ
        "tan_suat": tan_suat_tong,                                                                # cột tổng tần suất
    }).sort_values("tan_suat", ascending=False)                                                    # sắp xếp giảm dần

    # Ghép 2 bảng để có cả TF-IDF score lẫn tần suất cho mỗi cụm từ
    df_keyphrase_merged = df_tfidf_score.merge(                                                    # ghép 2 bảng
        df_count_score, on="keyphrase", how="inner"                                                # theo tên cụm từ, chỉ giữ cụm có ở cả 2
    )

    # --- 1.6 Lọc keyphrase chất lượng cao ---
    print("\nTiêu chí lọc keyphrase chất lượng cao:")                                              # in tiêu đề
    print("  (a) Ít nhất 2 từ (loại unigram quá chung chung)")                                    # giải thích tiêu chí a
    print("  (b) Tần suất xuất hiện >= 3 lần (loại cụm ngẫu nhiên)")                              # giải thích tiêu chí b
    print("  (c) TF-IDF score > median (chỉ giữ nửa trên)")                                       # giải thích tiêu chí c
    print("  (d) Lấy tối đa 150 keyphrase, xếp theo TF-IDF giảm dần")                             # giải thích tiêu chí d

    df_keyphrase_merged["so_tu"] = df_keyphrase_merged["keyphrase"].apply(                         # đếm số từ trong mỗi cụm
        lambda x: len(x.split())                                                                    # tách theo khoảng trắng và đếm
    )
    nguong_tfidf = df_keyphrase_merged["tfidf_score"].median()                                      # tính ngưỡng median TF-IDF
    print(f"  Ngưỡng TF-IDF median: {nguong_tfidf:.4f}")                                           # in ngưỡng

    df_keyphrase_loc = df_keyphrase_merged[                                                          # áp dụng 3 tiêu chí lọc
        (df_keyphrase_merged["so_tu"] >= 2) &                                                        # (a) ít nhất 2 từ
        (df_keyphrase_merged["tan_suat"] >= 3) &                                                     # (b) tần suất >= 3
        (df_keyphrase_merged["tfidf_score"] > nguong_tfidf)                                          # (c) TF-IDF > median
    ].sort_values("tfidf_score", ascending=False).head(150)                                          # (d) top 150, giảm dần

    KEYPHRASE_ATVSLD_AUTO = df_keyphrase_loc["keyphrase"].tolist()                                   # chuyển thành danh sách chuỗi

    # ============================================================
    # PHẦN 1.7: LỌC THỦ CÔNG KEYPHRASE — LOẠI BỎ CỤM TỪ KHÔNG MONG MUỐN
    # ============================================================
    # Sau khi TF-IDF tự động trích xuất, có thể còn lẫn cụm từ:
    #   - Quá chung chung (VD: "thực hiện", "ban hành")
    #   - Bị cắt sai ngữ cảnh (VD: "toàn vệ sinh" thay vì "an toàn vệ sinh")
    #   - Trùng lặp ý nghĩa (VD: "tai nạn" vs "tai nạn lao động")
    #   - Không phải thuật ngữ pháp lý (VD: tên riêng, số liệu...)
    #
    # CÓ 2 CÁCH LỌC THỦ CÔNG:
    #   Cách 1: DANH SÁCH ĐEN (blacklist) — liệt kê cụm từ CẦN LOẠI BỎ
    #   Cách 2: DANH SÁCH TRẮNG (whitelist) — liệt kê cụm từ CHẮC CHẮN GIỮ LẠI
    #
    # Bạn có thể dùng 1 trong 2 cách hoặc kết hợp cả 2.
    # ============================================================

    # --- CÁCH 1: DANH SÁCH ĐEN — những cụm từ CẦN LOẠI BỎ ---
    # Thêm/bớt tuỳ ý sau khi xem bảng keyphrase in ra ở Phần 1.7 bên trên
    DANH_SACH_DEN = [                                                                  # danh sách keyphrase cần loại bỏ
        # --- Cụm từ quá chung chung, không mang ý nghĩa pháp lý ---                                                                    # quá chung
        "sử_dụng lao_động",                                                                    # quá chung
        "lao_động bệnh nghề_nghiệp",
        "lao_động bệnh",
        "tai_nạn lao_động bệnh",
        "lao_động quy_định",
        "bảo_hiểm tai_nạn lao_động bệnh",
        "lao_động thương_binh xã_hội",
        "thương_binh xã_hội",
        "lao_động thương_binh",
        "quy_định 1",
        "bộ lao_động thương_binh",
        "bộ_trưởng bộ",
        "gây mất",
        "lao_động nơi",
        "yếu_tố nguy_hiểm yếu_tố",
        "nguy_hiểm yếu_tố",
        "lao_động làm_việc",
        "lao_động thực_hiện",
        "sự_cố kỹ_thuật gây",
        "kỹ_thuật gây mất",
        "kỹ thuật gây",
        "kỹ_thuật gây mất an_toàn vệ_sinh",
        "kỹ_thuật gây mất an_toàn vệ_sinh lao_động",
        "sự_cố kỹ_thuật gây mất",
        "kỹ_thuật gây mất an_toàn",
        "lao_động đối_với",
        "lao_động tai_nạn",
        "nguy_hiểm yếu_tố hại",
        "vệ_sinh lao_động nơi",
        "lao_động làm",
        "lao_động tai_nạn lao_động",
        "an_toàn vệ_sinh lao_động nơi",
        "chất yêu_cầu nghiêm_ngặt an_toàn vệ_sinh",
        "chất yêu_cầu",
        "chất yêu_cầu nghiêm_ngặt an_toàn vệ_sinh lao_động",
        "chất yêu_cầu nghiêm_ngặt an_toàn",
        "chất yêu_cầu nghiêm_ngặt",
        "lao_động trách_nhiệm",
        "bộ_trưởng bộ lao_động thương_binh",
        "bộ_trưởng bộ lao_động",
        "lao_động cơ_sở",
        "lao_động cấp",
        "lao_động sử_dụng",
        "ra tai_nạn",
        "làm_việc hợp_đồng",
        "vệ_sinh lao_động đối_với",
        "1 2 ",
        "vệ sinh lao động",
        "an toàn vệ sinh",
        "tai nạn lao động bệnh nghề nghiệp",
        "lao động có",
        "bảo đảm an toàn vệ sinh",
        "bảo hiểm tai nạn",
        "bảo hiểm tai nạn lao động bệnh nghề nghiệp",
        "có trách nhiệm",
        "máy thiết bị",
        "bộ lao động",
        "sản xuất kinh doanh",
        "suy giảm khả năng",
        "có hại",
        "huấn luyện an toàn",
        "huấn luyện an toàn vệ sinh",
        "yếu tố có",
        "thiết bị vật tư",
        "máy thiết bị vật tư",
        "mất an toàn vệ sinh",
        "công tác an toàn",
        "công tác an toàn vệ sinh",
        "điều tra tai nạn",
        "có yêu cầu",
        "đối với lao động",
        "có yêu cầu nghiêm ngặt",
        "yêu cầu nghiêm ngặt",
        "gây mất an toàn vệ sinh",
        "có liên quan",
        "lao động nơi làm việc",
        "sự cố kỹ thuật",
        "sự cố kỹ thuật gây mất an toàn",
        "sự cố kỹ thuật gây mất an toàn vệ sinh",
        "nghiêm ngặt an toàn",
        "có yêu cầu nghiêm ngặt an toàn",
        "nghiêm ngặt an toàn vệ sinh lao động",
        "có yêu cầu nghiêm ngặt an toàn vệ sinh",
        "yêu cầu nghiêm ngặt an toàn vệ sinh",
        "yêu cầu nghiêm ngặt an toàn",
        "nghiêm ngặt an toàn vệ sinh",
        "yêu cầu nghiêm ngặt an toàn vệ sinh lao động",
        "có yêu cầu nghiêm ngặt an toàn vệ sinh lao động",
        "cơ sở sản xuất",
        "biện pháp bảo đảm an toàn",
        "biện pháp bảo đảm an toàn vệ sinh",
        "biện pháp bảo đảm",
        "vật tư chất",
        "vệ sinh lao động nơi làm việc",
        "thiết bị vật tư chất",
        "máy thiết bị vật tư chất",
        "yếu tố nguy hiểm yếu tố có",
        "nguy hiểm yếu tố có hại",
        "nguy hiểm yếu tố có",
        "pháp luật an toàn vệ sinh",
        "pháp luật an toàn",
        "vệ sinh lao động quy định",
        "làm công tác",
        "an toàn vệ sinh lao động quy định",
        "lao động tổ chức",
        "sử dụng lao động có",
        "chất có",
        "chất có yêu cầu nghiêm ngặt",
        "chất có yêu cầu nghiêm ngặt an toàn vệ sinh",
        "chất có yêu cầu nghiêm ngặt an toàn",
        "chất có yêu cầu nghiêm ngặt an toàn vệ sinh lao động",
        "chất có yêu cầu",
        "an toàn vệ sinh lao động đối với",
        "pháp luật an toàn vệ sinh lao động",
        ""
        ""









                                                                                       # quá chung
        # --- Cụm từ bị cắt sai ngữ cảnh ---
        # "toàn vệ sinh",                                                              # VD: bị cắt thiếu "an" phía trước (bỏ comment nếu muốn loại)
        # --- Cụm từ trùng lặp (đã có phiên bản đầy đủ hơn) ---
        # "tai nạn",                                                                   # VD: đã có "tai nạn lao động" đầy đủ hơn
        # --- Thêm keyphrase cần loại bỏ vào đây ---
        # "abc xyz",
    ]

    # --- CÁCH 2: DANH SÁCH TRẮNG — những cụm từ BẮT BUỘC GIỮ LẠI ---
    # Dù TF-IDF có thể không trích xuất được (do min_df, max_df), vẫn thêm thủ công
    DANH_SACH_TRANG = [                                                                 # danh sách keyphrase bắt buộc giữ/thêm
        # --- Thuật ngữ dài mà TF-IDF có thể bỏ sót ---
        "máy thiết bị vật tư chất có yêu cầu nghiêm ngặt về an toàn vệ sinh lao động",  # 13 từ — thuật ngữ rất dài
        "nghề công việc nặng nhọc độc hại nguy hiểm",                                    # 7 từ — phân loại công việc
        "sự cố kỹ thuật gây mất an toàn vệ sinh lao động nghiêm trọng",                  # 10 từ — sự cố nghiêm trọng
        "sự cố kỹ thuật gây mất an toàn vệ sinh lao động",                                # 9 từ — sự cố chung
        "quỹ bảo hiểm tai nạn lao động bệnh nghề nghiệp",                                 # 8 từ — quỹ tài chính
        "nghề công việc đặc biệt nặng nhọc độc hại nguy hiểm",
        "yếu tố có hại",
        "người lao động",
        "người sử dụng lao động",
        "người lao động làm việc theo hợp đồng lao động",
        "người lao động làm việc không theo hợp đồng lao động",
        "người làm công tác an toàn vệ sinh lao động",
        "pháp luật về an toàn vệ sinh lao động",
                                                                                            # --- Thêm keyphrase bắt buộc giữ vào đây ---
        # "abc xyz",
    ]

    # --- ÁP DỤNG LỌC ---
    # Bước 1: Loại bỏ các cụm từ trong danh sách đen
    danh_sach_den_tach_tu = [tach_tu_cho_keyphrase(kp) for kp in DANH_SACH_DEN]         # tách từ danh sách đen để khớp chuẩn
    so_truoc_loc = len(KEYPHRASE_ATVSLD_AUTO)                                            # đếm số keyphrase trước khi lọc

    KEYPHRASE_ATVSLD_AUTO = [                                                             # lọc lại danh sách keyphrase
        kp for kp in KEYPHRASE_ATVSLD_AUTO                                                # giữ lại keyphrase
        if kp not in danh_sach_den_tach_tu                                                # nếu KHÔNG nằm trong danh sách đen
    ]

    so_da_loai = so_truoc_loc - len(KEYPHRASE_ATVSLD_AUTO)                                # đếm số keyphrase đã loại bỏ
    print(f"\n--- LỌC THỦ CÔNG ---")                                                      # in tiêu đề
    print(f"Đã loại bỏ {so_da_loai} keyphrase theo DANH SÁCH ĐEN")                        # in số đã loại

    # Bước 2: Thêm các cụm từ trong danh sách trắng (nếu chưa có)
    danh_sach_trang_tach_tu = [tach_tu_cho_keyphrase(kp) for kp in DANH_SACH_TRANG]       # tách từ danh sách trắng
    so_da_them = 0                                                                         # bộ đếm số keyphrase được thêm mới
    for kp_tach_tu in danh_sach_trang_tach_tu:                                              # duyệt qua danh sách trắng
        if kp_tach_tu not in KEYPHRASE_ATVSLD_AUTO:                                         # nếu keyphrase chưa có trong danh sách
            KEYPHRASE_ATVSLD_AUTO.append(kp_tach_tu)                                        # thêm vào cuối danh sách
            so_da_them += 1                                                                  # tăng bộ đếm

    print(f"Đã thêm {so_da_them} keyphrase theo DANH SÁCH TRẮNG")                          # in số đã thêm
    print(f"Tổng keyphrase sau lọc thủ công: {len(KEYPHRASE_ATVSLD_AUTO)}")                 # in tổng sau lọc

    # --- IN LẠI DANH SÁCH SAU KHI LỌC ---
    print(f"\n{'=' * 80}")                                                                   # đường kẻ
    print(f"DANH SÁCH KEYPHRASE SAU LỌC THỦ CÔNG ({len(KEYPHRASE_ATVSLD_AUTO)} cụm)")       # tiêu đề
    print(f"{'=' * 80}")                                                                     # đường kẻ
    for stt, kp in enumerate(KEYPHRASE_ATVSLD_AUTO, start=1):                                 # duyệt từng keyphrase
        kp_dep = kp.replace("_", " ")                                                         # bỏ gạch dưới để hiển thị
        so_tu = len(kp.split())                                                                # đếm số từ
        nguon = "⚪ tự động" if kp not in danh_sach_trang_tach_tu else "🟢 thủ công"           # đánh dấu nguồn gốc
        print(f"  {stt:>4}. {kp_dep:<65} ({so_tu} từ) {nguon}")                                # in STT + keyphrase + số từ + nguồn

    # --- 1.7 In kết quả keyphrase đã trích xuất ---
    print(f"\n{'=' * 80}")                                                                            # đường kẻ
    print(f"KẾT QUẢ: ĐÃ TRÍCH XUẤT {len(KEYPHRASE_ATVSLD_AUTO)} KEYPHRASE TỰ ĐỘNG")                  # in tổng số
    print(f"{'=' * 80}")                                                                               # đường kẻ
    print(f"\n{'STT':<5} {'Keyphrase':<65} {'TF-IDF':>10} {'Tần suất':>10} {'Số từ':>7}")             # header bảng
    print("-" * 100)                                                                                    # đường kẻ
    for stt, (_, row) in enumerate(df_keyphrase_loc.iterrows(), start=1):                               # duyệt từng keyphrase
        kp_hien_thi = row["keyphrase"].replace("_", " ")                                                # bỏ gạch dưới để hiển thị đẹp
        print(f"{stt:<5} {kp_hien_thi:<65} {row['tfidf_score']:>10.4f} "                                # in STT + keyphrase + TF-IDF
              f"{int(row['tan_suat']):>10} {int(row['so_tu']):>7}")                                      # in tần suất + số từ

    # Giải thích ý nghĩa cột TF-IDF score
    print(f"\n{'=' * 80}")                                                                               # đường kẻ
    print("GIẢI THÍCH CỘT 'TF-IDF':")                                                                    # tiêu đề
    print("  - Là TỔNG điểm TF-IDF của cụm từ đó trên TOÀN BỘ 606 chunk")                                # giải thích dòng 1
    print("  - Điểm CAO = cụm từ VỪA phổ biến trong nhiều chunk,")                                        # giải thích dòng 2
    print("    VỪA có tính phân biệt (không phải hư từ xuất hiện ở mọi nơi)")                              # giải thích dòng 3
    print("  - VD: 'tai nạn lao động' có TF-IDF cao vì xuất hiện nhiều")                                   # ví dụ 1
    print("    nhưng không phải ở MỌI chunk (chỉ ở các Điều liên quan TNLĐ)")                               # giải thích tại sao cao
    print("  - VD: 'và' có TF-IDF ≈ 0 vì xuất hiện ở MỌI chunk -> IDF ≈ 0")                                # ví dụ 2

    # ============================================================
    # PHẦN 2: TỪ ĐIỂN ĐỒNG NGHĨA / VIẾT TẮT PHÁP LÝ NGÀNH ATVSLĐ
    # ============================================================

    TU_DONG_NGHIA = {                                                                                        # từ điển viết tắt -> thuật ngữ chuẩn
        "tnlđ": "tai nạn lao động",                                                                          # viết tắt phổ biến ngành
        "bnn": "bệnh nghề nghiệp",                                                                           # viết tắt phổ biến ngành
        "atvslđ": "an toàn vệ sinh lao động",                                                                # viết tắt tên luật
        "nsdlđ": "người sử dụng lao động",                                                                   # viết tắt chủ thể sử dụng lao động
        "nlđ": "người lao động",                                                                              # viết tắt chủ thể người lao động
        "bị tai nạn": "tai nạn lao động",                                                                     # khẩu ngữ -> thuật ngữ chuẩn
        "bị bệnh nghề": "bệnh nghề nghiệp",                                                                  # khẩu ngữ -> thuật ngữ chuẩn
        "công nhân": "người lao động",                                                                         # khẩu ngữ -> thuật ngữ chuẩn
        "công ty": "người sử dụng lao động",                                                                  # khẩu ngữ -> thuật ngữ chuẩn
        "sếp": "người sử dụng lao động",                                                                      # khẩu ngữ -> thuật ngữ chuẩn
        "tiền bồi thường": "bồi thường tai nạn lao động",                                                     # khẩu ngữ -> thuật ngữ chuẩn
        "phụ cấp độc hại": "phụ cấp trách nhiệm",                                                             # khẩu ngữ -> thuật ngữ chuẩn
    }

    def mo_rong_dong_nghia(cau_hoi: str) -> str:                                                               # hàm mở rộng câu hỏi bằng đồng nghĩa
        """Bổ sung thuật ngữ chuẩn vào câu hỏi nếu phát hiện viết tắt/khẩu ngữ.
        VD: 'NLĐ bị TNLĐ' -> 'NLĐ bị TNLĐ người lao động tai nạn lao động'"""
        cau_hoi_thuong = cau_hoi.lower()                                                                       # chuyển câu hỏi về chữ thường
        phan_bo_sung = []                                                                                       # danh sách thuật ngữ chuẩn sẽ nối thêm
        for tu_tat, tu_chuan in TU_DONG_NGHIA.items():                                                          # duyệt từng cặp viết tắt -> chuẩn
            if tu_tat in cau_hoi_thuong:                                                                        # nếu viết tắt có trong câu hỏi
                phan_bo_sung.append(tu_chuan)                                                                   # thêm thuật ngữ chuẩn vào danh sách
        if phan_bo_sung:                                                                                         # nếu có ít nhất 1 thuật ngữ cần bổ sung
            return cau_hoi + " " + " ".join(phan_bo_sung)                                                       # nối thêm vào cuối câu hỏi gốc
        return cau_hoi                                                                                            # trả về câu hỏi không đổi nếu không tìm thấy

    # ============================================================
    # PHẦN 3: ĐẶC TẢ DẠNG QUY ĐỊNH (NHÃN NGỮ NGHĨA CHO TỪNG CHUNK)
    # ============================================================

    DAC_TA_DANG_QUY_DINH = {                                                                                     # từ điển: nhãn dạng quy định -> từ khoá tín hiệu
        "dinh_nghia":  ["được hiểu là", "là việc", "là hành vi",                                                  # tín hiệu cho điều khoản giải thích từ ngữ
                        "trong luật này", "là tai nạn", "là bệnh phát sinh"],
        "quyen":       ["có quyền", "được quyền", "được hưởng",                                                   # tín hiệu cho quy định về quyền
                        "được trả", "được cung cấp", "được đào tạo",
                        "được bảo đảm"],
        "nghia_vu":    ["có nghĩa vụ", "có trách nhiệm", "phải thực hiện",                                       # tín hiệu cho quy định về nghĩa vụ
                        "phải báo cáo", "chấp hành", "phải đóng"],
        "dieu_kien":   ["điều kiện", "trường hợp", "khi đáp ứng",                                                 # tín hiệu cho quy định điều kiện
                        "khi xảy ra", "đủ điều kiện"],
        "thu_tuc":     ["hồ sơ", "trình tự", "thủ tục", "nộp tại",                                               # tín hiệu cho quy định thủ tục
                        "cơ quan có thẩm quyền", "lập biên bản",
                        "đăng ký", "kê khai"],
        "thoi_han":    ["trong thời hạn", "chậm nhất", "kể từ ngày",                                              # tín hiệu cho quy định thời hạn
                        "trong vòng", "không quá"],
        "che_tai":     ["xử phạt", "bị thu hồi", "cưỡng chế",                                                    # tín hiệu cho quy định chế tài
                        "bồi thường thiệt hại", "trốn đóng", "chậm đóng",
                        "bị đình chỉ", "bị xử lý"],
    }

    def gan_nhan_dang_quy_dinh(noi_dung: str) -> str:                                                             # hàm gán nhãn dạng quy định cho 1 chunk
        """Phân loại chunk thuộc dạng: định nghĩa / quyền / nghĩa vụ / điều kiện /
        thủ tục / thời hạn / chế tài. Không khớp mẫu nào -> 'khac'."""
        noi_dung_thuong = str(noi_dung).lower()                                                                    # chuyển nội dung về chữ thường
        for nhan, tu_khoa_list in DAC_TA_DANG_QUY_DINH.items():                                                    # duyệt từng nhãn và danh sách từ khoá
            if any(tk in noi_dung_thuong for tk in tu_khoa_list):                                                   # nếu bất kỳ từ khoá nào xuất hiện
                return nhan                                                                                          # trả về nhãn tương ứng
        return "khac"                                                                                                # không khớp -> nhãn mặc định

    # ============================================================
    # PHẦN 4: GÁN KEYPHRASE + DẠNG QUY ĐỊNH VÀO df_chunks VÀ LƯU CSV
    # ============================================================

    print(f"\n{'=' * 80}")                                                                                           # đường kẻ
    print("BƯỚC 5: GÁN KEYPHRASE + DẠNG QUY ĐỊNH VÀO TỪNG CHUNK")                                                   # tiêu đề
    print(f"{'=' * 80}")                                                                                              # đường kẻ

    # --- 4.1 Gán keyphrase khớp cho từng dòng ---
    def tim_keyphrase_trong_dong(noi_dung: str) -> str:                                                               # hàm tìm keyphrase trong 1 dòng
        """Tìm tất cả keyphrase (đã trích xuất tự động bằng TF-IDF) xuất hiện
        trong nội dung 1 chunk. Trả về chuỗi cách nhau bằng dấu | để lưu CSV."""
        noi_dung_tach_tu = tach_tu_cho_keyphrase(noi_dung)                                                            # tách từ nội dung chunk
        cac_kp_khop = []                                                                                               # danh sách keyphrase khớp
        for kp in KEYPHRASE_ATVSLD_AUTO:                                                                               # duyệt qua từng keyphrase tự động
            if kp in noi_dung_tach_tu:                                                                                 # nếu keyphrase có trong nội dung đã tách từ
                kp_dep = kp.replace("_", " ")                                                                          # bỏ gạch dưới để hiển thị đẹp
                cac_kp_khop.append(kp_dep)                                                                             # thêm vào danh sách kết quả
        return "|".join(cac_kp_khop) if cac_kp_khop else ""                                                            # nối bằng | hoặc trả về chuỗi rỗng

    df_chunks["keyphrase"] = df_chunks["noi_dung"].apply(tim_keyphrase_trong_dong)                                     # gán cột keyphrase cho mỗi dòng
    df_chunks["dang_quy_dinh"] = df_chunks["noi_dung"].apply(gan_nhan_dang_quy_dinh)                                   # gán cột dạng quy định cho mỗi dòng

    # --- 4.2 In kết quả kiểm tra ---
    print(f"\n--- PHÂN BỐ DẠNG QUY ĐỊNH ---")                                                                          # tiêu đề
    print(df_chunks["dang_quy_dinh"].value_counts().to_string())                                                        # đếm và in từng loại

    so_dong_co_kp = (df_chunks["keyphrase"] != "").sum()                                                                # đếm số dòng có keyphrase
    print(f"\n--- THỐNG KÊ KEYPHRASE ---")                                                                              # tiêu đề
    print(f"Số dòng có keyphrase: {so_dong_co_kp} / {len(df_chunks)}"                                                   # in số dòng có keyphrase
          f" ({so_dong_co_kp/len(df_chunks)*100:.1f}%)")                                                                 # in tỷ lệ phần trăm

    # In 10 dòng mẫu có keyphrase dài nhất (thể hiện n-gram lớn hoạt động)
    print(f"\n--- 10 DÒNG MẪU CÓ KEYPHRASE DÀI NHẤT ---")                                                               # tiêu đề
    df_co_kp = df_chunks[df_chunks["keyphrase"] != ""].copy()                                                            # lọc dòng có keyphrase
    df_co_kp["do_dai_kp"] = df_co_kp["keyphrase"].apply(                                                                 # tính độ dài keyphrase dài nhất
        lambda x: max(len(kp.split()) for kp in x.split("|")) if x else 0                                                # đếm số từ của keyphrase dài nhất trong dòng
    )
    for _, row in df_co_kp.nlargest(10, "do_dai_kp")[                                                                     # lấy 10 dòng có keyphrase dài nhất
        ["full_citation", "dang_quy_dinh", "keyphrase"]].iterrows():                                                      # chọn 3 cột cần hiển thị
        print(f"  [{row['dang_quy_dinh']:<12}] {row['full_citation']}")                                                   # in nhãn + trích dẫn
        print(f"    Keyphrase: {row['keyphrase']}")                                                                        # in keyphrase
        print()                                                                                                            # dòng trống

    # --- 4.3 Lưu lại file CSV chính (ghi đè, bổ sung 2 cột mới) ---
    file_path_csv = duong_dan_chunks               # đường dẫn file CSV chính
    df_chunks = _chuan_hoa_kieu_cot_truoc_khi_ghi(df_chunks)                                    # giữ dieu_id/khoan_id là số nguyên khi ghi
    df_chunks.to_csv(file_path_csv, index=False, encoding='utf-8-sig')                                                     # lưu CSV kèm 2 cột mới
    print(f"\n✅ Đã lưu file CSV bổ sung keyphrase + dạng quy định tại:")                                                  # thông báo
    print(f"   {file_path_csv}")                                                                                            # in đường dẫn

    # --- 4.4 Lưu bảng keyphrase riêng để tham khảo/báo cáo ---
    file_path_keyphrase = duong_dan_keyphrase             # đường dẫn file keyphrase
    df_keyphrase_loc.to_csv(file_path_keyphrase, index=False, encoding='utf-8-sig')                                         # lưu bảng keyphrase ra CSV
    print(f"✅ Đã lưu bảng {len(df_keyphrase_loc)} keyphrase tại:")                                                         # thông báo
    print(f"   {file_path_keyphrase}")                                                                                       # in đường dẫn

    # --- 4.5 In cấu trúc cột cuối cùng ---
    print(f"\n{'=' * 80}")                                                                                                   # đường kẻ
    print("CẤU TRÚC CỘT CỦA df_chunks SAU KHI LÀM GIÀU")                                                                   # tiêu đề
    print(f"{'=' * 80}")                                                                                                      # đường kẻ
    print(df_chunks.columns.tolist())                                                                                          # in danh sách tên cột
    print(f"Tổng số dòng: {len(df_chunks)}")                                                                                  # in tổng số dòng
    print(f"Tổng số keyphrase trích xuất: {len(KEYPHRASE_ATVSLD_AUTO)}")                                                      # in tổng số keyphrase

    if luu:
        os.makedirs(os.path.dirname(duong_dan_keyphrase), exist_ok=True)
    return df_chunks, df_keyphrase_loc, KEYPHRASE_ATVSLD_AUTO


def main() -> None:
    p = argparse.ArgumentParser(description="Trích keyphrase pháp lý từ chunks bằng TF-IDF n-gram")
    p.add_argument("--chunks", default=None, help="đường dẫn CSV chunks (mặc định: data/processed/law_dataset_chunks.csv)")
    p.add_argument("--keyphrase", dest="keyphrase", default=None, help="nơi ghi CSV keyphrase")
    a = p.parse_args()

    duong_dan_chunks = a.chunks or DUONG_DAN_CHUNKS
    df = pd.read_csv(duong_dan_chunks, encoding="utf-8-sig")
    print(f"📄 Đọc {len(df)} chunks từ {duong_dan_chunks}")

    df, df_kp, ds_kp = lam_giau_keyphrase(df, duong_dan_chunks, a.keyphrase)

    df = _chuan_hoa_kieu_cot_truoc_khi_ghi(df)                                  # giữ dieu_id/khoan_id là số nguyên khi ghi
    df.to_csv(duong_dan_chunks, index=False, encoding="utf-8-sig")
    print(f"💾 Đã ghi lại chunks ({df.shape[1]} cột) -> {duong_dan_chunks}")
    print(f"💾 Đã ghi {len(df_kp)} keyphrase -> {a.keyphrase or DUONG_DAN_KEYPHRASE}")


if __name__ == "__main__":
    main()
