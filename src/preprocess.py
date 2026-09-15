"""
src/preprocess.py — Đọc văn bản luật (.docx) -> chuẩn hoá -> tách thành chunks.
Kết quả mong đợi: 606 chunks
    cap_do:  khoan 358 | diem 201 | khoan_bo_sung 36 | dieu_intro 11
Ghi ra: data/processed/law_dataset_chunks.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import docx2txt
import pandas as pd

GOC_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THU_MUC_RAW = os.path.join(GOC_REPO, "data", "raw")
DUONG_DAN_CHUNKS_MAC_DINH = os.path.join(GOC_REPO, "data", "processed", "law_dataset_chunks.csv")


# ============================================================
# PHẦN 1 — ĐỌC FILE .DOCX  (ô 8 trong notebook)
# ============================================================
def extract_all_text_from_docx(file_path):
    try:
        # docx2txt tự động đọc toàn bộ paragraph và table theo đúng thứ tự hiển thị
        full_text = docx2txt.process(file_path)
        return full_text
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None

# ============================================================
# PHẦN 2 — CHUẨN HOÁ VĂN BẢN BẰNG REGEX  (ô 9 trong notebook)
# ============================================================
#Sử dụng regex để chuẩn hóa các ký tự, xử lý các lỗi font chữ thường gặp...

def standardize_text(text): #Chuẩn hóa văn bản, xử lý các lỗi font và ký tự đặc biệt
    if not text: #Kiểm tra nếu đầu vào là chuỗi rỗng, None hoặc False
        return "" #Trả về chuỗi rỗng ngay lập tức để tránh lỗi khi xử lý

    text = re.sub(r'[\x82\x84\x85\x91\x92\x93\x94\x96\x97]', ' ', text) #Thay thế các ký tự bị lỗi font thường gặp thành dấu cách

    text = re.sub(r'[ \t]+', ' ', text).strip()#Thay thế nhiều khoảng trắng/tab liền nhau thành 1 khoảng trắng, và xóa khoảng trắng ở 2 đầu chuỗi

    text = re.sub(r'[^\w\s\.\,\;\n]', ' ', text)#(Tùy chọn) Bỏ comment dòng này nếu muốn loại bỏ sạch các ký tự đặc biệt, chỉ giữ lại chữ, số, khoảng trắng và các dấu . , ;

    return text #Trả về kết quả chuỗi văn bản đã được chuẩn hóa hoàn tất

# ============================================================
# PHẦN 3 — TÁCH VĂN BẢN LUẬT THÀNH CHUNKS  (ô 10 trong notebook)
# ============================================================


def parse_law_text_to_chunks(text):  # ham chinh, nhan van ban da chuan hoa, tra ve DataFrame co cau truc
    danh_sach_doan = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]  # tach van ban thanh tung doan, dua vao dong trong (docx2txt phan cach doan bang \n\n)

    MAU_CHUONG = re.compile(r"^Chương\s+([IVXLCDM]+)$")                   # mau nhan dien dong "Chương I", "Chương II"... KHONG co dau cham hay ten tren cung dong
    MAU_DIEU = re.compile(r"^Điều\s+(\d+)\.\s*(.+)$")                     # mau nhan dien dong "Điều 6. Ten dieu", nhom 1 la so dieu, nhom 2 la ten
    MAU_KHOAN = re.compile(r"^(\d+)\.\s*(.+)$")                           # mau nhan dien mot Khoan dang "1. Noi dung", dau cham khong bi standardize_text xoa
    MAU_DIEM = re.compile(r"^([a-zđươ]{1,2})\s{2,}(.+)$", re.IGNORECASE)  # mau nhan dien Diem: sau khi mat dau ")" thi "a) Noi dung" thanh "a  Noi dung" (2 khoang trang)

    rows = []                       # list chua cac dict, moi dict la 1 dong ket qua cuoi cung (1 Khoan/Diem/cau mo dau)
    chuong_id = None                # so hieu Chuong dang xu ly (vd "I", "II"), cap nhat khi gap tieu de Chuong moi
    chuong_ten = None               # ten day du cua Chuong dang xu ly (vd "QUY ĐỊNH CHUNG")
    cho_ten_chuong = False          # co bao hieu doan tiep theo chinh la TEN cua Chuong vua gap
    dieu_id = None                  # so hieu Dieu dang xu ly (vd "6"), cap nhat khi gap tieu de Dieu moi
    dieu_ten = None                 # ten day du cua Dieu dang xu ly
    khoan_id = None                 # so hieu Khoan dang xu ly, dung de gan Diem (a, b, c) vao dung Khoan cha
    intro_buf = []                  # list tam gom cac cau mo dau cua 1 Dieu (truoc khi gap Khoan dau tien)

    def citation(khoan=None, diem=None):                  # ham con tao chuoi trich dan day du dua tren Dieu/Khoan/Diem hien tai
        s = f"Điều {dieu_id}"                               # phan bat buoc: luon co so Dieu
        if khoan is not None:                               # neu co Khoan thi them vao
            s += f", Khoản {khoan}"                          # noi them "Khoan X"
        if diem is not None:                                # neu co Diem thi them vao
            s += f", Điểm {diem}"                            # noi them "Diem x"
        return f"{s} - Luật An toàn, vệ sinh lao động số 84/2015/QH13"  # ghep ten day du bo luat vao cuoi

    def flush_intro():                                     # ham con day cau mo dau da gom duoc thanh 1 dong "dieu_intro"
        if intro_buf:                                        # chi thuc hien neu thuc su co noi dung can ghi
            rows.append({                                     # them 1 dong moi vao ket qua
                'chuong_id': chuong_id, 'chuong_ten': chuong_ten,      # gan Chuong hien tai
                'dieu_id': dieu_id, 'dieu_ten': dieu_ten,              # gan Dieu hien tai
                'khoan_id': None, 'diem_id': None,                     # dong intro khong thuoc Khoan/Diem nao
                'cap_do': 'dieu_intro',                                # danh dau cap do
                'noi_dung': ' '.join(intro_buf).strip(),               # noi cac cau mo dau thanh 1 doan hoan chinh
                'full_citation': citation(),                           # trich dan chi voi so Dieu
            })
            intro_buf.clear()                                # don sach list tam cho Dieu tiep theo

    for doan in danh_sach_doan:                            # duyet qua tung doan theo dung thu tu trong file

        m = MAU_CHUONG.match(doan)                          # kiem tra doan co phai tieu de Chuong khong
        if m:                                               # neu dung la tieu de Chuong moi
            flush_intro()                                    # ghi not cau mo dau cua Dieu truoc do (neu con)
            chuong_id = m.group(1)                           # cap nhat so hieu Chuong
            cho_ten_chuong = True                            # bat co cho ten Chuong o dong ke tiep
            dieu_id = None                                   # reset Dieu vi sang Chuong moi
            khoan_id = None                                  # reset Khoan vi sang Chuong moi
            continue                                         # sang doan tiep theo

        if cho_ten_chuong:                                  # neu dong truoc la tieu de Chuong, dong nay la ten Chuong
            chuong_ten = doan.strip()                        # luu ten Chuong
            cho_ten_chuong = False                           # tat co
            continue                                         # sang doan tiep theo

        m = MAU_DIEU.match(doan)                            # kiem tra doan co phai tieu de Dieu khong
        if m:                                               # neu dung la tieu de Dieu moi
            flush_intro()                                    # ghi not cau mo dau cua Dieu TRUOC do (neu con)
            dieu_id = m.group(1)                             # cap nhat so hieu Dieu
            dieu_ten = m.group(2).strip()                    # cap nhat ten Dieu
            khoan_id = None                                  # reset Khoan vi sang Dieu moi
            continue                                         # sang doan tiep theo

        if dieu_id is None:                                 # neu chua tung gap Dieu nao (dang o phan mo dau van ban)
            continue                                         # bo qua hoan toan doan nay

        m_diem = MAU_DIEM.match(doan)                       # kiem tra doan co phai mot Diem khong
        m_khoan = MAU_KHOAN.match(doan)                     # kiem tra doan co phai mot Khoan khong

        if (m_diem or m_khoan) and intro_buf:               # neu sap ghi Khoan/Diem DAU TIEN ma con cau mo dau chua ghi
            flush_intro()                                    # ghi not cau mo dau NGAY LUC NAY, dat truoc Khoan 1

        if m_diem and khoan_id is not None:                 # neu la Diem VA da co Khoan cha dang xu ly
            rows.append({                                     # them dong moi dai dien cho Diem nay
                'chuong_id': chuong_id, 'chuong_ten': chuong_ten,
                'dieu_id': dieu_id, 'dieu_ten': dieu_ten,
                'khoan_id': khoan_id,                          # gan Khoan cha
                'diem_id': m_diem.group(1).strip(),            # ky hieu Diem (a, b, c...)
                'cap_do': 'diem',                              # danh dau cap do Diem
                'noi_dung': m_diem.group(2).strip(),           # noi dung cua Diem
                'full_citation': citation(khoan_id, m_diem.group(1).strip()),  # trich dan day du Dieu-Khoan-Diem
            })
            continue                                         # sang doan tiep theo

        if m_khoan:                                         # neu la mot dong Khoan moi
            khoan_id = m_khoan.group(1)                      # cap nhat so Khoan hien tai
            rows.append({                                     # them dong moi dai dien cho Khoan nay
                'chuong_id': chuong_id, 'chuong_ten': chuong_ten,
                'dieu_id': dieu_id, 'dieu_ten': dieu_ten,
                'khoan_id': khoan_id, 'diem_id': None,         # Khoan khong co Diem rieng
                'cap_do': 'khoan',                             # danh dau cap do Khoan
                'noi_dung': m_khoan.group(2).strip(),          # noi dung cua Khoan
                'full_citation': citation(khoan_id),           # trich dan Dieu-Khoan
            })
            continue                                         # sang doan tiep theo

        if khoan_id is None:                                # neu doan nay khong khop Khoan/Diem VA chua co Khoan nao
            intro_buf.append(doan.strip())                   # day la cau mo dau cua Dieu, gom tam lai
        else:                                                # neu da co Khoan truoc do roi
            rows.append({                                     # day la cau bo sung/ket luan cua Dieu, van ghi thanh dong rieng
                'chuong_id': chuong_id, 'chuong_ten': chuong_ten,
                'dieu_id': dieu_id, 'dieu_ten': dieu_ten,
                'khoan_id': khoan_id, 'diem_id': None,
                'cap_do': 'khoan_bo_sung',                     # danh dau day la cau bo sung
                'noi_dung': doan.strip(),                      # noi dung nguyen van cua cau bo sung
                'full_citation': citation(),                   # trich dan chi voi so Dieu
            })

    flush_intro()                                           # sau vong lap, ghi not cau mo dau con lai cua Dieu CUOI CUNG
    df = pd.DataFrame(rows)                                 # chuyen toan bo list dict thanh DataFrame
    df.insert(0, 'id', range(1, len(df) + 1))               # them cot id danh so thu tu tang dan tu 1

    df['ten_van_ban'] = 'Luật An toàn, vệ sinh lao động 2015'     # cot metadata: ten van ban, giong nhau moi dong
    df['ngay_ban_hanh'] = 'ngày 25 tháng 06 năm 2015'             # cot metadata: ngay ban hanh
    df['co_quan_ban_hanh'] = 'Quốc Hội'                            # cot metadata: co quan ban hanh
    df['so_hieu'] = 'Luật số: 84/2015/QH13'                        # cot metadata: so hieu van ban
    return df                                               # tra ve DataFrame hoan chinh

# ============================================================
# PHẦN 4 — HÀM TIỆN ÍCH + CHẠY TỪ DÒNG LỆNH
# ============================================================
def tim_file_docx(duong_dan: str | None = None) -> str:
    """Tìm file .docx đầu tiên trong data/raw nếu không truyền đường dẫn."""
    if duong_dan:
        return duong_dan
    ung_vien = sorted(glob.glob(os.path.join(THU_MUC_RAW, "*.docx")))
    if not ung_vien:
        raise FileNotFoundError(
            f"Không tìm thấy file .docx nào trong {THU_MUC_RAW}.\n"
            "Hãy tải file luật gốc về thư mục đó (xem README.md)."
        )
    return ung_vien[0]


def tao_chunks(duong_dan_docx: str | None = None,
               duong_dan_luu: str | None = None,
               luu: bool = True) -> pd.DataFrame:
    """Chạy trọn quy trình: .docx -> chuẩn hoá -> DataFrame chunks (và ghi CSV)."""
    duong_dan_docx = tim_file_docx(duong_dan_docx)
    print(f"📄 Đọc file luật: {duong_dan_docx}")

    van_ban_tho = extract_all_text_from_docx(duong_dan_docx)
    if not van_ban_tho:
        raise ValueError(f"Không đọc được nội dung file: {duong_dan_docx}")

    van_ban = standardize_text(van_ban_tho)
    print(f"🧹 Sau chuẩn hoá: {len(van_ban):,} ký tự")

    df = parse_law_text_to_chunks(van_ban)
    print(f"✅ Đã tạo {len(df)} chunks")
    print(df["cap_do"].value_counts().to_string())

    if luu:
        duong_dan_luu = duong_dan_luu or DUONG_DAN_CHUNKS_MAC_DINH
        os.makedirs(os.path.dirname(duong_dan_luu), exist_ok=True)
        df.to_csv(duong_dan_luu, index=False, encoding="utf-8-sig")
        print(f"💾 Đã lưu: {duong_dan_luu}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Đọc .docx và tách thành chunks theo Điều/Khoản/Điểm")
    p.add_argument("--docx", default=None, help="đường dẫn file .docx (mặc định: tự dò trong data/raw)")
    p.add_argument("--ra", dest="ra", default=None, help="nơi ghi CSV (mặc định: data/processed/law_dataset_chunks.csv)")
    p.add_argument("--khong-luu", action="store_true", help="chỉ in thống kê, không ghi file")
    a = p.parse_args()
    tao_chunks(a.docx, a.ra, luu=not a.khong_luu)


if __name__ == "__main__":
    main()
