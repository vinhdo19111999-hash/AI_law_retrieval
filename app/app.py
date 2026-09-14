"""Chatbot tra cứu Luật An toàn, vệ sinh lao động 2015 — giao diện Streamlit.

FILE THẬT trong repo (không sinh ra từ notebook nữa).

--------------------------------------------------------------------
CÁCH CHẠY
--------------------------------------------------------------------
1) Trên máy tính (sau khi đã clone repo):
       pip install -r requirements.txt
       streamlit run app/app.py

2) Trên Google Colab:
       %cd /content/NLP_law_retrieval
       !pip install -q streamlit
       !streamlit run app/app.py --server.port 8501 &>/content/log.txt &
       # rồi mở tunnel (ngrok / cloudflared) trỏ vào cổng 8501

3) Đổi model nhẹ hơn (nếu máy yếu, RAM thấp):
       đặt biến môi trường MODEL_E5=intfloat/multilingual-e5-base
       hoặc sửa trực tiếp TÊN_MODEL ở dưới.

--------------------------------------------------------------------
DỮ LIỆU CẦN CÓ
--------------------------------------------------------------------
    data/processed/law_dataset_chunks.csv      <- bắt buộc
    models/e5_embeddings.npy                   <- tự tạo lần đầu, dùng lại lần sau
"""
import glob
import os
import re

import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==================== CẤU HÌNH ====================
GOC_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TÊN_MODEL = os.environ.get("MODEL_E5", "intfloat/multilingual-e5-large")
# RAM yếu thì đổi thành: "intfloat/multilingual-e5-base" (nhẹ hơn ~2 lần, chất lượng gần tương đương)

DUONG_DAN_EMB = os.environ.get(
    "DUONG_DAN_EMB", os.path.join(GOC_REPO, "models", "e5_embeddings.npy")
)
DUNG_CACHE_EMB = True      # True = lưu embedding ra .npy, lần sau chạy lại rất nhanh
NGUONG_MAC_DINH = 0.25     # ngưỡng tương đồng tối thiểu để coi là "có liên quan"
TOP_K_MAC_DINH = 3

# Ba biến toàn cục, được gán khi app khởi động (xem phần "GIAO DIỆN STREAMLIT" ở cuối file)
df_chunks = None           # bảng 606 chunks luật
corpus_embeds = None       # ma trận embedding của toàn bộ corpus
model_emb = None           # model SentenceTransformer đã nạp
# ==================================================


def tim_file_chunks():
    """Tìm file chunks luật. Tự dò để không phụ thuộc đường dẫn cứng."""
    ung_vien = [
        os.environ.get("DUONG_DAN_CHUNKS", ""),
        os.path.join(GOC_REPO, "data", "processed", "law_dataset_chunks.csv"),
        "data/processed/law_dataset_chunks.csv",
        "/content/NLP_law_retrieval/data/processed/law_dataset_chunks.csv",
        "/content/drive/MyDrive/Colab Notebooks/NLP/law_retrieval/data/processed/law_dataset_chunks.csv",
    ]
    for d in ung_vien:
        if d and os.path.exists(d):
            return d
    tim = sorted(glob.glob("/content/**/data/processed/law_dataset_chunks.csv", recursive=True))
    if tim:
        return tim[0]
    raise FileNotFoundError(
        "Không tìm thấy data/processed/law_dataset_chunks.csv.\n"
        "Hãy chạy ô 7 + ô 8 trong notebook (hoặc ô LUU-VA-PUSH) để tạo file này, "
        "rồi đặt biến môi trường DUONG_DAN_CHUNKS nếu file nằm chỗ khác."
    )


# ==================== NẠP DỮ LIỆU + MODEL (cache) ====================
@st.cache_resource(show_spinner="Đang đọc dữ liệu luật...")
def nap_du_lieu():
    duong_dan = tim_file_chunks()
    df = pd.read_csv(duong_dan, encoding="utf-8-sig")

    # Ghép câu dẫn của Khoản vào các Điểm con -> giúp tìm kiếm chính xác hơn
    van_ban_tim_kiem = []
    for _, row in df.iterrows():
        if row["cap_do"] in ("diem", "khoan_bo_sung") and pd.notna(row["khoan_id"]):
            khoan_cha = df[(df["dieu_id"] == row["dieu_id"]) &
                           (df["khoan_id"] == row["khoan_id"]) &
                           (df["cap_do"] == "khoan")]
            cau_dan = khoan_cha.iloc[0]["noi_dung"] if not khoan_cha.empty else ""
            van_ban_tim_kiem.append(f"{cau_dan} {row['noi_dung']}")
        else:
            van_ban_tim_kiem.append(row["noi_dung"])
    df["van_ban_tim_kiem"] = van_ban_tim_kiem
    return df, duong_dan


@st.cache_resource(show_spinner="Đang nạp model embedding (lần đầu có thể mất 1–3 phút)...")
def nap_model_va_embedding(duong_dan_chunks: str):
    import numpy as np

    df = st.session_state.get("_df_chunks")
    thiet_bi = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(TÊN_MODEL, device=thiet_bi)
    corpus = df["van_ban_tim_kiem"].astype(str).tolist()

    # Dùng lại embedding đã lưu nếu đúng số dòng (nhanh hơn nhiều lần)
    if DUNG_CACHE_EMB and os.path.exists(DUONG_DAN_EMB):
        try:
            emb = np.load(DUONG_DAN_EMB)
            if emb.shape[0] == len(corpus):
                return model, emb
        except Exception:
            pass

    emb = model.encode(corpus, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    if DUNG_CACHE_EMB:
        os.makedirs(os.path.dirname(DUONG_DAN_EMB), exist_ok=True)
        np.save(DUONG_DAN_EMB, emb)
    return model, emb


def get_embedding(texts):
    if isinstance(texts, str):
        texts = [texts]
    return model_emb.encode(texts, convert_to_numpy=True, show_progress_bar=False)


# ==================== TỪ ĐIỂN ĐỒNG NGHĨA ====================
TU_DONG_NGHIA = {
    "tnlđ": "tai nạn lao động", "bnn": "bệnh nghề nghiệp",
    "atvslđ": "an toàn vệ sinh lao động",
    "nsdlđ": "người sử dụng lao động", "nlđ": "người lao động",
    "bị tai nạn": "tai nạn lao động", "công nhân": "người lao động",
    "công ty": "người sử dụng lao động",
}


def mo_rong_dong_nghia(cau_hoi: str) -> str:
    cau_hoi_thuong = cau_hoi.lower()
    bo_sung = [tu_chuan for tu_tat, tu_chuan in TU_DONG_NGHIA.items() if tu_tat in cau_hoi_thuong]
    return cau_hoi + " " + " ".join(bo_sung) if bo_sung else cau_hoi


# ==================== BÀI TOÁN 0: METADATA ====================
def tra_loi_metadata(cau_hoi):
    cau_hoi_thuong = cau_hoi.lower()
    metadata = df_chunks.iloc[0]
    if "hiệu lực" in cau_hoi_thuong:
        dong = df_chunks[df_chunks["dieu_ten"].str.contains("Hiệu lực", case=False, na=False)]
        if not dong.empty:
            return f"📖 {dong.iloc[0]['full_citation']}\n\n{dong.iloc[0]['noi_dung']}"
        return "Không tìm thấy."
    if "cơ quan ban hành" in cau_hoi_thuong or "do ai ban hành" in cau_hoi_thuong:
        return (f"**Cơ quan ban hành:** {metadata['co_quan_ban_hanh']}  \n"
                f"*(Nguồn: {metadata['ten_van_ban']}, {metadata['so_hieu']})*")
    if "ngày ban hành" in cau_hoi_thuong:
        return (f"**Ngày ban hành:** {metadata['ngay_ban_hanh']}  \n"
                f"*(Nguồn: {metadata['ten_van_ban']}, {metadata['so_hieu']})*")
    if "số hiệu" in cau_hoi_thuong:
        return f"**Số hiệu:** {metadata['so_hieu']}"
    return None


# ==================== BÀI TOÁN 1: TRẢ VỀ NGUYÊN VĂN ====================
def bai_toan_1(cau_hoi):
    m_dieu = re.search(r"[Đđ]i[eề]u\s*(?:số\s*)?(\d+)", cau_hoi)
    if not m_dieu:
        return None
    so_dieu = int(m_dieu.group(1))
    m_khoan = re.search(r"[Kk]ho[ảa]n\s*(?:số\s*)?(\d+)", cau_hoi)
    so_khoan = int(m_khoan.group(1)) if m_khoan else None
    m_diem = re.search(r"[Đđ]i[ểe]m\s*([a-zđươA-ZĐƯƠ])\b", cau_hoi)
    ky_diem = m_diem.group(1).lower() if m_diem else None

    ket_qua = df_chunks[df_chunks["dieu_id"] == so_dieu]
    if ket_qua.empty:
        return None
    if so_khoan is not None:
        ket_qua = ket_qua[(ket_qua["khoan_id"] == so_khoan) |
                          (ket_qua["khoan_id"].isna() & (ket_qua["cap_do"] == "dieu_intro"))]
    if ky_diem is not None:
        ket_qua = ket_qua[(ket_qua["diem_id"] == ky_diem) | (ket_qua["cap_do"] == "khoan")]
    if ket_qua.empty:
        return f"Không tìm thấy Điều {so_dieu}"

    lines = []
    for _, row in ket_qua.sort_values("id").iterrows():
        if row["cap_do"] == "diem":
            lines.append(f"  {row['diem_id']}) {row['noi_dung']}")
        elif row["cap_do"] == "khoan":
            lines.append(f"{int(row['khoan_id'])}. {row['noi_dung']}")
        else:
            lines.append(row["noi_dung"])
    citation = (ket_qua.iloc[0]["full_citation"] if len(ket_qua) == 1
                else f"Điều {so_dieu} - {ket_qua.iloc[0]['ten_van_ban']}")
    return f"📖 **{citation}**\n\n" + "\n".join(lines)


# ==================== BÀI TOÁN 2: TRA CỨU NGỮ NGHĨA ====================
def bai_toan_2(cau_hoi, top_k=3, nguong=0.25):
    cau_hoi_mo = mo_rong_dong_nghia(cau_hoi)
    q_emb = get_embedding(cau_hoi_mo)
    sim = cosine_similarity(q_emb, corpus_embeds)[0]
    top_idx = sim.argsort()[::-1][:top_k]
    results = []
    for idx in top_idx:
        if sim[idx] < nguong:
            break
        dong = df_chunks.iloc[idx]
        results.append({
            "diem": float(sim[idx]),
            "citation": dong["full_citation"],
            "noi_dung": dong["noi_dung"],
            "cap_do": dong["cap_do"],
            "dieu_id": dong["dieu_id"],
            "khoan_id": dong["khoan_id"],
            "keyphrase": dong.get("keyphrase", ""),
        })
    return results


def mo_rong_khoan(results):
    """Nếu kết quả là Khoản, in kèm các Điểm con cho đầy đủ."""
    new_results = []
    for r in results:
        dong_goc = df_chunks[df_chunks["full_citation"] == r["citation"]]
        if not dong_goc.empty and dong_goc.iloc[0]["cap_do"] == "khoan":
            dieu_id = dong_goc.iloc[0]["dieu_id"]
            khoan_id = dong_goc.iloc[0]["khoan_id"]
            cac_diem = df_chunks[(df_chunks["dieu_id"] == dieu_id) &
                                 (df_chunks["khoan_id"] == khoan_id) &
                                 (df_chunks["cap_do"] == "diem")]
            if not cac_diem.empty:
                noi_dung = r["noi_dung"] + "\n" + "\n".join(
                    f"   {d['diem_id']}) {d['noi_dung']}" for _, d in cac_diem.iterrows())
                r = {**r, "noi_dung": noi_dung}
        new_results.append(r)
    return new_results


def bai_toan_2_full(cau_hoi, top_k=3, nguong=0.25):
    raw = bai_toan_2(cau_hoi, top_k=top_k, nguong=nguong)
    if not raw:
        return "Không tìm thấy thông tin liên quan."
    raw = mo_rong_khoan(raw)
    output = []
    for i, r in enumerate(raw, 1):
        output.append(f"**{i}.** 📖 **{r['citation']}** "
                      f"*(độ liên quan: {r['diem']:.4f})*\n\n{r['noi_dung']}")
    return "\n\n---\n\n".join(output)


# ==================== PIPELINE 3 TẦNG ====================
def chatbot_tra_loi(cau_hoi, top_k=3, nguong=0.25):
    meta = tra_loi_metadata(cau_hoi)            # Tầng 0 — metadata văn bản
    if meta:
        return "Thông tin văn bản", meta
    b1 = bai_toan_1(cau_hoi)                    # Tầng 1 — hỏi thẳng Điều/Khoản
    if b1:
        return "Tra cứu Điều/Khoản cụ thể", b1
    b2 = bai_toan_2_full(cau_hoi, top_k=top_k, nguong=nguong)   # Tầng 2 — ngữ nghĩa
    if "Không tìm thấy" in b2:
        return ("Ngoài phạm vi",
                "Xin lỗi, tôi không tìm thấy thông tin liên quan trong Luật ATVSLĐ 2015.")
    return "Tra cứu ngữ nghĩa (Embedding)", b2


# ==================== GIAO DIỆN STREAMLIT ====================
st.set_page_config(page_title="Chatbot Luật ATVSLĐ 2015", page_icon="⚖️", layout="centered")
st.title("⚖️ Chatbot tra cứu Luật An toàn, vệ sinh lao động 2015")
st.caption("Luật số 84/2015/QH13 — hỏi trực tiếp theo Điều/Khoản hoặc bằng ngôn ngữ tự nhiên.")

try:
    df_chunks, duong_dan_chunks = nap_du_lieu()
except FileNotFoundError as loi:
    st.error(str(loi))
    st.stop()
st.session_state["_df_chunks"] = df_chunks

with st.spinner("Đang nạp model embedding..."):
    model_emb, corpus_embeds = nap_model_va_embedding(duong_dan_chunks)

with st.sidebar:
    st.header("⚙️ Cài đặt tra cứu")
    top_k = st.slider("Số kết quả trả về (top_k)", 1, 10, TOP_K_MAC_DINH)
    nguong = st.slider("Ngưỡng tương đồng tối thiểu", 0.0, 0.9, NGUONG_MAC_DINH, 0.05)
    st.caption("Ngưỡng cao hơn → chỉ trả lời khi chắc chắn; ngưỡng thấp hơn → trả lời nhiều hơn nhưng dễ sai.")

    st.divider()
    st.header("ℹ️ Thông tin")
    st.markdown(f"""
    - **Tên luật:** {df_chunks['ten_van_ban'].iloc[0]}
    - **Số hiệu:** {df_chunks['so_hieu'].iloc[0]}
    - **Cơ quan ban hành:** {df_chunks['co_quan_ban_hanh'].iloc[0]}
    - **Số Điều:** {df_chunks['dieu_id'].nunique()} / 93
    - **Số chunk:** {len(df_chunks)}
    - **Model:** `{TÊN_MODEL.split('/')[-1]}`
    """)

    st.header("💡 Câu hỏi mẫu")
    for v in [
        "Cơ quan ban hành Luật là gì?",
        "Điều 38 khoản 4 nói gì?",
        "Người lao động có quyền từ chối làm việc nguy hiểm không?",
        "Ai có trách nhiệm điều tra tai nạn lao động?",
        "Bồi thường khi suy giảm 81% khả năng lao động là bao nhiêu?",
    ]:
        if st.button(v, use_container_width=True):
            st.session_state["cau_hoi_mau"] = v

    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state["lich_su"] = []

if "lich_su" not in st.session_state:
    st.session_state["lich_su"] = []

for role, msg in st.session_state["lich_su"]:
    with st.chat_message(role):
        st.markdown(msg)

cau_hoi = st.chat_input("Nhập câu hỏi về Luật ATVSLĐ 2015...")
cau_hoi_mau = st.session_state.pop("cau_hoi_mau", None)
cau_hoi_final = cau_hoi or cau_hoi_mau

if cau_hoi_final:
    st.session_state["lich_su"].append(("user", cau_hoi_final))
    with st.chat_message("user"):
        st.markdown(cau_hoi_final)
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu..."):
            loai, dap_an = chatbot_tra_loi(cau_hoi_final, top_k=top_k, nguong=nguong)
        st.caption(f"🔎 {loai}")
        st.markdown(dap_an)
        if loai.startswith("Tra cứu ngữ nghĩa"):
            with st.expander("Xem cách hệ thống tìm ra kết quả này"):
                for r in bai_toan_2(cau_hoi_final, top_k=top_k, nguong=nguong):
                    st.write(f"`{r['diem']:.4f}` — **{r['citation']}** "
                             f"(cấp: {r['cap_do']})")
                    if isinstance(r.get("keyphrase"), str) and r["keyphrase"]:
                        st.caption(f"Từ khoá: {r['keyphrase'].replace('|', ' · ')}")
    st.session_state["lich_su"].append(
        ("assistant", f"🔎 *{loai}*\n\n{dap_an}"))

