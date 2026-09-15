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
import os
import sys

import streamlit as st

# Trỏ về thư mục gốc repo để import được package src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import retrieval

TÊN_MODEL = retrieval.TÊN_MODEL
NGUONG_MAC_DINH = retrieval.NGUONG_MAC_DINH
TOP_K_MAC_DINH = retrieval.TOP_K_MAC_DINH


















@st.cache_resource(show_spinner="Đang nạp dữ liệu và model embedding (lần đầu có thể mất 1–3 phút)...")
def tai_du_lieu_va_model():
    retrieval.khoi_tao(nap_embedding=True)          # nạp df_chunks + model E5 + embedding
    return retrieval.df_chunks, retrieval.model_emb, retrieval.corpus_embeds
       





































































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

# ==================== GIAO DIỆN STREAMLIT ====================
st.set_page_config(page_title="Chatbot Luật ATVSLĐ 2015", page_icon="⚖️", layout="centered")
st.title("⚖️ Chatbot tra cứu Luật An toàn, vệ sinh lao động 2015")
st.caption("Luật số 84/2015/QH13 — hỏi trực tiếp theo Điều/Khoản hoặc bằng ngôn ngữ tự nhiên.")

try:
    df_chunks, model_emb, corpus_embeds = tai_du_lieu_va_model()
except FileNotFoundError as loi:
    st.error(str(loi))
    st.stop()
       
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
            loai, dap_an = retrieval.chatbot_tra_loi(cau_hoi_final, top_k=top_k, nguong=nguong)
        st.caption(f"🔎 {loai}")
        st.markdown(dap_an)
        if loai.startswith("Tra cứu ngữ nghĩa"):
            with st.expander("Xem cách hệ thống tìm ra kết quả này"):
                for r in retrieval.bai_toan_2(cau_hoi_final, top_k=top_k, nguong=nguong):
                    st.write(f"`{r['diem']:.4f}` — **{r['citation']}** "
                             f"(cấp: {r['cap_do']})")
                    if isinstance(r.get("keyphrase"), str) and r["keyphrase"]:
                        st.caption(f"Từ khoá: {r['keyphrase'].replace('|', ' · ')}")
    st.session_state["lich_su"].append(
        ("assistant", f"🔎 *{loai}*\n\n{dap_an}"))

