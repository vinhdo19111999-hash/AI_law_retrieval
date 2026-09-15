"""
src/evaluation.py — Chấm chất lượng tra cứu trên bộ 60 câu hỏi (Hit@1/3/5 + MRR).
Kết quả in ra: Hit@1 / Hit@3 / Hit@5 / MRR + bảng theo Chương + bảng theo độ khó
File xuất ra: 6 ảnh PNG + 5 bảng CSV trong reports/hinh-anh/
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

GOC_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUONG_DAN_TEST_SET_MAC_DINH = os.path.join(GOC_REPO, "data", "test_questions.csv")
DUONG_DAN_CHUNKS_MAC_DINH = os.path.join(GOC_REPO, "data", "processed", "law_dataset_chunks.csv")
THU_MUC_ANH_MAC_DINH = os.path.join(GOC_REPO, "reports", "hinh-anh")
DUONG_DAN_TEST_SET_GITHUB = ("https://raw.githubusercontent.com/"
                             "vinhdo19111999-hash/NLP_law_retrieval/main/"
                             "data/test_questions.csv")
KHO_ANH_MAC_DINH = 150
CAC_CHI_SO = ["Hit@1", "Hit@3", "Hit@5", "MRR"]


def chay_danh_gia(ham_truy_van, ten_phuong_phap="E5 embedding", df_chunks=None,
                  duong_dan_test_set=DUONG_DAN_TEST_SET_MAC_DINH,
                  thu_muc_anh=THU_MUC_ANH_MAC_DINH,
                  top_k=5, nguong=0.0, chay_so_sanh_tfidf=True,
                  kho_anh=KHO_ANH_MAC_DINH, tai_anh_ve_may=False):
    """Chấm 1 hàm tra cứu trên bộ câu hỏi, in bảng và ghi ảnh/CSV vào thu_muc_anh."""
    # ------------------------- 1. ĐỌC BỘ CÂU HỎI --------------------------------
    try:
        duong_dan_test = duong_dan_test_set          # biến từ ô cấu hình đường dẫn
    except NameError:
        duong_dan_test = "data/test_questions.csv"

    if os.path.exists(duong_dan_test):
        print(f"📄 Đọc bộ câu hỏi: {duong_dan_test}")
        df_test = pd.read_csv(duong_dan_test, encoding="utf-8-sig")
    else:
        print("⚠️  Không thấy file trong repo -> tải trực tiếp từ GitHub (nhánh main).")
        df_test = pd.read_csv(DUONG_DAN_TEST_SET_GITHUB, encoding="utf-8-sig")

    for cot in ["cau_hoi", "loai_truy_van", "dap_an_dung"]:
        if cot not in df_test.columns:
            raise SystemExit(f"❌ File câu hỏi thiếu cột '{cot}'. Cột hiện có: {list(df_test.columns)}")

    # dap_an_dung PHẢI là list, vì khop() viết 'for d in dap_an'
    BO_CAU_HOI_CHUAN = [
        {"cau_hoi": r.cau_hoi,
         "loai_truy_van": r.loai_truy_van,
         "dap_an_dung": str(r.dap_an_dung).split("|")}
        for r in df_test.itertuples()
    ]
    print(f"✅ Đã nạp {len(BO_CAU_HOI_CHUAN)} câu hỏi "
          f"(loại 'ngu_nghia': {sum(1 for m in BO_CAU_HOI_CHUAN if m['loai_truy_van'] == 'ngu_nghia')})")

    # ------------------------- 2. HÀM CHẤM ĐIỂM ---------------------------------
    def khop(citation, dap_an):
        """True nếu bất kỳ chuỗi nào trong dap_an xuất hiện trong citation."""
        if not dap_an:
            return False
        return any(d.lower() in str(citation).lower() for d in dap_an)


    def danh_gia(ham_truy_van, ten_phuong_phap):
        """Chạy cả bộ câu hỏi qua 1 hàm tra cứu, trả về DataFrame kết quả."""
        ket_qua = []
        for mau in BO_CAU_HOI_CHUAN:
            if mau["loai_truy_van"] != "ngu_nghia":
                continue
            t0 = time.time()
            raw = ham_truy_van(mau["cau_hoi"], top_k=top_k, nguong=nguong)
            citations = [r["full_citation"] for r in raw]
            hang = 0                                    # 1..top_k, 0 = không có trong top_k
            for rank, c in enumerate(citations, 1):
                if khop(c, mau["dap_an_dung"]):
                    hang = rank
                    break
            ket_qua.append({
                "phuong_phap": ten_phuong_phap,
                "cau_hoi": mau["cau_hoi"],
                "dap_an_dung": " | ".join(mau["dap_an_dung"]),
                "top1_citation": citations[0] if citations else "N/A",
                "hang_dung": hang,
                "Hit@1": 1 if hang == 1 else 0,
                "Hit@3": 1 if 1 <= hang <= 3 else 0,
                "Hit@5": 1 if 1 <= hang <= 5 else 0,
                "MRR": (1.0 / hang) if hang else 0.0,
                "thoi_gian": round(time.time() - t0, 4),
            })
        return pd.DataFrame(ket_qua)


    # ------------------------- 3. CHẠY ĐÁNH GIÁ (E5) ----------------------------
    print("\n" + "=" * 78)
    print("ĐÁNH GIÁ TRA CỨU NGỮ NGHĨA (EMBEDDING + COSINE SIMILARITY)")
    print("=" * 78)
    print(f"top_k = {top_k}   |   nguong = {nguong}\n")

    df_eval = danh_gia(ham_truy_van, ten_phuong_phap)

    # Chốt an toàn: tránh lỗi KeyError 'Hit@1' như bản cũ (df rỗng -> không có cột)
    if df_eval.empty:
        raise SystemExit(
            "❌ Không có câu nào được chấm.\n"
            "   Kiểm tra: cột loai_truy_van trong CSV có đúng giá trị 'ngu_nghia' không?"
        )

    # Ghép thêm Chương / Độ khó để chia bảng
    df_eval = df_eval.merge(df_test[["cau_hoi", "nhom_chu_de", "do_kho"]],
                            on="cau_hoi", how="left")

    print("=" * 78)
    print(f"KẾT QUẢ TRÊN {len(df_eval)} CÂU HỎI")
    print("=" * 78)
    for c in CAC_CHI_SO:
        print(f"  {c:<8} {df_eval[c].mean():.4f}")
    print(f"  {'Tỉ lệ có đáp án đúng trong top-' + str(top_k):<38} "
          f"{(df_eval['hang_dung'] > 0).mean():.4f}")

    bang_chuong = (df_eval.groupby("nhom_chu_de")[CAC_CHI_SO].mean().round(4)
                   .join(df_eval.groupby("nhom_chu_de").size().rename("so_cau")))
    bang_kho = (df_eval.groupby("do_kho")[CAC_CHI_SO].mean().round(4)
                .join(df_eval.groupby("do_kho").size().rename("so_cau")))

    print("\n--- CHIA THEO CHƯƠNG ---")
    print(bang_chuong.to_string())
    print("\n--- CHIA THEO ĐỘ KHÓ ---")
    print(bang_kho.to_string())

    print("\n--- 10 CÂU HỆ THỐNG LÀM SAI (Hit@1 = 0) ---")
    for _, r in df_eval[df_eval["Hit@1"] == 0].head(10).iterrows():
        print(f"❌ {r['cau_hoi'][:72]}")
        print(f"   đáp án đúng : {r['dap_an_dung']}")
        print(f"   máy trả về  : {r['top1_citation']}")

    # ------------------------- 4. SO SÁNH VỚI BASELINE TF-IDF -------------------
    df_tfidf = None
    if chay_so_sanh_tfidf:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as _cos

            _corpus = df_chunks["van_ban_tim_kiem"].astype(str).tolist()
            _vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
            _X = _vec.fit_transform(_corpus)

            def _tra_cuu_tfidf(cau_hoi, top_k=5, nguong=0.0):
                _sim = _cos(_vec.transform([cau_hoi]), _X)[0]
                ket = []
                for idx in _sim.argsort()[::-1][:top_k]:
                    if _sim[idx] < nguong:
                        continue
                    d = df_chunks.iloc[idx]
                    ket.append({"diem_tuong_dong": float(_sim[idx]),
                                "full_citation": d["full_citation"],
                                "noi_dung": d["noi_dung"],
                                "cap_do": d["cap_do"]})
                return ket

            print("\n" + "=" * 78)
            print("SO SÁNH PHƯƠNG PHÁP: TF-IDF (baseline) vs E5 EMBEDDING")
            print("=" * 78)
            df_tfidf = danh_gia(_tra_cuu_tfidf, "TF-IDF")
            df_tfidf = df_tfidf.merge(df_test[["cau_hoi", "nhom_chu_de", "do_kho"]],
                                      on="cau_hoi", how="left")

            bang_ss = pd.DataFrame({
                "TF-IDF": [df_tfidf[c].mean() for c in CAC_CHI_SO],
                ten_phuong_phap: [df_eval[c].mean() for c in CAC_CHI_SO],
            }, index=CAC_CHI_SO).round(4)
            bang_ss["Chênh lệch"] = (bang_ss[ten_phuong_phap] - bang_ss["TF-IDF"]).round(4)
            print(bang_ss.to_string())
            print(f"\nThời gian tra cứu trung bình: "
                  f"TF-IDF {df_tfidf['thoi_gian'].mean()*1000:.1f} ms/câu  |  "
                  f"E5 {df_eval['thoi_gian'].mean()*1000:.1f} ms/câu")
            chenh = bang_ss.loc["Hit@1", "Chênh lệch"]
            if chenh > 0:
                print(f"👉 E5 tốt hơn TF-IDF {chenh:+.4f} ở Hit@1 "
                      f"-> dùng được làm luận điểm chính trong báo cáo.")
            else:
                print(f"👉 Ở bộ đề này E5 chưa hơn TF-IDF ({chenh:+.4f} ở Hit@1). "
                      "Hãy kiểm tra lại: model có được nạp đúng không, có dùng tiền tố "
                      "'query: ' / 'passage: ' chưa (e5 yêu cầu tiền tố này).")
        except Exception as e:
            print(f"\n⚠️  Bỏ qua phần so sánh TF-IDF vì lỗi: {e}")

    # ------------------------- 5. BIỂU ĐỒ --------------------------------------
    _co_font = {f.name for f in font_manager.fontManager.ttflist}
    for _ten in ["DejaVu Sans", "Liberation Sans", "Noto Sans", "Arial", "FreeSans"]:
        if _ten in _co_font:
            plt.rcParams["font.family"] = _ten
            break
    plt.rcParams["axes.unicode_minus"] = False
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    THU_MUC_ANH = thu_muc_anh
    os.makedirs(THU_MUC_ANH, exist_ok=True)


    def luu_anh(fig, ten_file):
        duong_dan = os.path.join(THU_MUC_ANH, ten_file)
        fig.savefig(duong_dan, dpi=kho_anh, bbox_inches="tight")
        print(f"   💾 {duong_dan}")
        return duong_dan


    def luu_bang(df, ten_file, **kw):
        duong_dan = os.path.join(THU_MUC_ANH, ten_file)
        df.to_csv(duong_dan, encoding="utf-8-sig", **kw)
        print(f"   💾 {duong_dan}")


    MAU = {"chinh": "#2E5AAC", "phu": "#7BA7D7", "dat": "#4C9A52", "sai": "#C1554C"}

    print("\n" + "=" * 78)
    print("VẼ BIỂU ĐỒ")
    print("=" * 78)

    # --- Biểu đồ 1: 4 chỉ số tổng -------------------------------------------------
    gia_tri = [df_eval[c].mean() for c in CAC_CHI_SO]
    fig1, ax = plt.subplots(figsize=(7.5, 4.6))
    cot = ax.bar(CAC_CHI_SO, gia_tri, color=[MAU["chinh"]] * 3 + ["#E0A33B"], width=0.6)
    ax.bar_label(cot, fmt="%.4f", padding=3, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Giá trị (0 → 1)")
    ax.set_title(f"Chất lượng tra cứu trên {len(df_eval)} câu hỏi\n"
                 f"(multilingual-e5, top_k={top_k})", fontsize=12)
    plt.tight_layout()
    luu_anh(fig1, "bieu-do-1-chi-so-tong.png")

    # --- Biểu đồ 2: theo Chương --------------------------------------------------
    fig2, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(bang_chuong))
    rong = 0.38
    ax.barh(y - rong / 2, bang_chuong["Hit@1"], rong, label="Hit@1", color=MAU["chinh"])
    ax.barh(y + rong / 2, bang_chuong["Hit@5"], rong, label="Hit@5", color=MAU["phu"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i}  (n={int(n)})"
                        for i, n in zip(bang_chuong.index, bang_chuong["so_cau"])], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Giá trị (0 → 1)")
    ax.set_title("Chất lượng tra cứu theo Chương của Luật ATVSLĐ 2015", fontsize=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    luu_anh(fig2, "bieu-do-2-theo-chuong.png")

    # --- Biểu đồ 3: phân bố thứ hạng của kết quả đúng ----------------------------
    nhan_hang = [f"Hạng {i}" for i in range(1, top_k + 1)] + ["Không có\ntrong top-%d" % top_k]
    dem_hang = [int((df_eval["hang_dung"] == i).sum()) for i in range(1, top_k + 1)]
    dem_hang.append(int((df_eval["hang_dung"] == 0).sum()))
    mau_hang = [MAU["dat"]] + ["#8FBF8A"] * (top_k - 1) + [MAU["sai"]]
    fig3, ax = plt.subplots(figsize=(8.4, 4.6))
    cot3 = ax.bar(nhan_hang, dem_hang, color=mau_hang, width=0.62)
    ax.bar_label(cot3, fmt="%d câu", padding=3, fontsize=10)
    ax.set_ylim(0, max(dem_hang) * 1.22 if max(dem_hang) else 1)
    ax.set_ylabel("Số câu hỏi")
    ax.set_title("Kết quả đúng nằm ở thứ hạng nào trong 5 kết quả trả về?", fontsize=12)
    plt.tight_layout()
    fig3.subplots_adjust(bottom=0.22)
    luu_anh(fig3, "bieu-do-3-phan-bo-thu-hang.png")

    # --- Biểu đồ 4: theo độ khó --------------------------------------------------
    thu_tu_kho = [k for k in ["de", "trung_binh", "kho"] if k in bang_kho.index]
    bk = bang_kho.loc[thu_tu_kho]
    x = np.arange(len(bk))
    rong2 = 0.2
    fig4, ax = plt.subplots(figsize=(8.4, 4.8))
    for k, (ten, mau) in enumerate(zip(CAC_CHI_SO, [MAU["chinh"], MAU["phu"], "#A7C4E4", "#E0A33B"])):
        cot4 = ax.bar(x + (k - 1.5) * rong2, bk[ten], rong2, label=ten, color=mau)
        ax.bar_label(cot4, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={int(n)})" for t, n in zip(bk.index, bk["so_cau"])])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Giá trị (0 → 1)")
    ax.set_title("Chất lượng tra cứu theo độ khó của câu hỏi", fontsize=12)
    ax.legend(ncol=4, loc="upper center", fontsize=9)
    plt.tight_layout()
    luu_anh(fig4, "bieu-do-4-theo-do-kho.png")

    # --- Biểu đồ 5: so sánh TF-IDF vs E5 (nếu có) --------------------------------
    if df_tfidf is not None:
        fig_ss, ax = plt.subplots(figsize=(8.4, 4.8))
        x2 = np.arange(len(CAC_CHI_SO))
        gt_tfidf = [df_tfidf[c].mean() for c in CAC_CHI_SO]
        cot_a = ax.bar(x2 - 0.2, gt_tfidf, 0.4, label="TF-IDF (baseline)", color="#9AA5B1")
        cot_b = ax.bar(x2 + 0.2, gia_tri, 0.4, label=ten_phuong_phap, color=MAU["chinh"])
        ax.bar_label(cot_a, fmt="%.3f", fontsize=9, padding=2)
        ax.bar_label(cot_b, fmt="%.3f", fontsize=9, padding=2)
        ax.set_xticks(x2)
        ax.set_xticklabels(CAC_CHI_SO)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Giá trị (0 → 1)")
        ax.set_title("So sánh hai phương pháp tra cứu trên cùng bộ 60 câu hỏi", fontsize=12)
        ax.legend(loc="upper right")
        plt.tight_layout()
        luu_anh(fig_ss, "bieu-do-5-so-sanh-tfidf-e5.png")

    # --- Biểu đồ gộp 2x2: chèn thẳng vào báo cáo ---------------------------------
    fig5, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(CAC_CHI_SO, gia_tri, color=[MAU["chinh"]] * 3 + ["#E0A33B"], width=0.6)
    axes[0, 0].bar_label(axes[0, 0].containers[0], fmt="%.3f", padding=3)
    axes[0, 0].set_ylim(0, 1.12)
    axes[0, 0].set_title("(a) Chỉ số tổng hợp")

    yy = np.arange(len(bang_chuong))
    axes[0, 1].barh(yy - rong / 2, bang_chuong["Hit@1"], rong, label="Hit@1", color=MAU["chinh"])
    axes[0, 1].barh(yy + rong / 2, bang_chuong["Hit@5"], rong, label="Hit@5", color=MAU["phu"])
    axes[0, 1].set_yticks(yy)
    axes[0, 1].set_yticklabels(bang_chuong.index, fontsize=9)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlim(0, 1.05)
    axes[0, 1].set_title("(b) Theo Chương")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(nhan_hang, dem_hang, color=mau_hang, width=0.62)
    axes[1, 0].bar_label(axes[1, 0].containers[0], fmt="%d", padding=3)
    axes[1, 0].set_title("(c) Phân bố thứ hạng của kết quả đúng")

    for k, (ten, mau) in enumerate(zip(CAC_CHI_SO, [MAU["chinh"], MAU["phu"], "#A7C4E4", "#E0A33B"])):
        axes[1, 1].bar(x + (k - 1.5) * rong2, bk[ten], rong2, label=ten, color=mau)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(bk.index)
    axes[1, 1].set_ylim(0, 1.15)
    axes[1, 1].set_title("(d) Theo độ khó")
    axes[1, 1].legend(ncol=4, fontsize=8)

    fig5.suptitle("Đánh giá hệ thống tra cứu Luật An toàn, vệ sinh lao động 2015",
                  fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    luu_anh(fig5, "tong-hop-4-bieu-do.png")
    plt.show()

    # ------------------------- 6. LƯU BẢNG RA CSV -------------------------------
    print("\n" + "=" * 78)
    print("LƯU BẢNG RA CSV")
    print("=" * 78)
    luu_bang(df_eval, "ket-qua-danh-gia.csv", index=False)
    luu_bang(bang_chuong, "bang-theo-chuong.csv")
    luu_bang(bang_kho, "bang-theo-do-kho.csv")
    if df_tfidf is not None:
        luu_bang(bang_ss, "bang-so-sanh-phuong-phap.csv")
        luu_bang(df_tfidf, "ket-qua-tfidf.csv", index=False)

    # ------------------------- 7. TÙY CHỌN: TẢI ẢNH VỀ MÁY ----------------------
    if tai_anh_ve_may:
        try:
            from google.colab import files
        except ImportError:
            print("⚠️  --tai-anh chỉ dùng được khi chạy trên Colab — bỏ qua.")
            files = None
        for ten in ([] if files is None else ["bieu-do-1-chi-so-tong.png", "bieu-do-2-theo-chuong.png",
                    "bieu-do-3-phan-bo-thu-hang.png", "bieu-do-4-theo-do-kho.png",
                    "bieu-do-5-so-sanh-tfidf-e5.png", "tong-hop-4-bieu-do.png"]):
            duong_dan = os.path.join(THU_MUC_ANH, ten)
            if os.path.exists(duong_dan):
                files.download(duong_dan)

    # ------------------------- 8. GỢI Ý BƯỚC TIẾP -------------------------------
    print("\n" + "-" * 78)
    print("BƯỚC TIẾP THEO:")
    print("   1. Chèn ảnh tong-hop-4-bieu-do.png vào mục KẾT QUẢ & THẢO LUẬN của báo cáo.")
    print("   2. Nếu có ảnh bieu-do-5-so-sanh-tfidf-e5.png -> chèn thêm vào mục 'So sánh phương pháp'.")
    print("   3. Chép 4 con số Hit@1 / Hit@3 / Hit@5 / MRR vào bảng kết quả.")
    print("   4. Ghi lại 3 câu sai nhiều nhất + giải thích vì sao (đây là phần được điểm cao).")
    print("   5. Muốn thử ngưỡng chống câu hỏi ngoài miền: chạy lại với --nguong 0.25.")
    print("-" * 78)

# ============================================================
# CHẠY TỪ DÒNG LỆNH
# ============================================================
def _ham_tfidf_builder(df_chunks):
    """Tạo hàm tra cứu baseline TF-IDF trên chính corpus (để so sánh với E5)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cos

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X = vec.fit_transform(df_chunks["van_ban_tim_kiem"].astype(str).tolist())

    def tra_cuu(cau_hoi, top_k=5, nguong=0.0):
        sim = _cos(vec.transform([cau_hoi]), X)[0]
        ket = []
        for idx in sim.argsort()[::-1][:top_k]:
            if sim[idx] < nguong:
                continue
            d = df_chunks.iloc[idx]
            ket.append({"diem_tuong_dong": float(sim[idx]),
                        "full_citation": d["full_citation"], "noi_dung": d["noi_dung"],
                        "cap_do": d["cap_do"]})
        return ket
    return tra_cuu


def main() -> None:
    p = argparse.ArgumentParser(description="Chấm chất lượng tra cứu trên bộ 60 câu hỏi")
    p.add_argument("--test-set", dest="test_set", default=DUONG_DAN_TEST_SET_MAC_DINH)
    p.add_argument("--chunks", default=DUONG_DAN_CHUNKS_MAC_DINH)
    p.add_argument("--ra", dest="ra", default=THU_MUC_ANH_MAC_DINH, help="thư mục ghi ảnh + CSV")
    p.add_argument("--top-k", dest="top_k", type=int, default=5)
    p.add_argument("--nguong", type=float, default=0.0)
    p.add_argument("--chi-tfidf", dest="chi_tfidf", action="store_true",
                   help="chỉ chạy baseline TF-IDF (không cần model E5, chạy vài giây)")
    p.add_argument("--tai-anh", dest="tai_anh", action="store_true", help="tự tải ảnh về máy (chỉ trên Colab)")
    a = p.parse_args()

    os.environ["DUONG_DAN_CHUNKS"] = a.chunks
    try:
        from .retrieval import bai_toan_2_tra_cuu_ngu_nghia, khoi_tao, nap_du_lieu
    except ImportError:                                   # khi chạy trực tiếp: python src/evaluation.py
        from retrieval import bai_toan_2_tra_cuu_ngu_nghia, khoi_tao, nap_du_lieu

    df_chunks, _ = nap_du_lieu()                          # đã có cột van_ban_tim_kiem
    print(f"📄 Corpus: {len(df_chunks)} chunks")

    if a.chi_tfidf:
        chay_danh_gia(_ham_tfidf_builder(df_chunks), "TF-IDF", df_chunks,
                      a.test_set, a.ra, a.top_k, a.nguong,
                      chay_so_sanh_tfidf=False, tai_anh_ve_may=a.tai_anh)
    else:
        khoi_tao(a.chunks)
        chay_danh_gia(bai_toan_2_tra_cuu_ngu_nghia,
                      "E5 embedding", df_chunks, a.test_set, a.ra, a.top_k, a.nguong,
                      chay_so_sanh_tfidf=True, tai_anh_ve_may=a.tai_anh)


if __name__ == "__main__":
    main()
