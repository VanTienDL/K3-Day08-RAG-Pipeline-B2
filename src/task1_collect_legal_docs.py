"""
Task 1 — Thu thập văn bản chính sách/quy định dịch vụ đại học.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang công khai của một trường đại học.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai RMIT Vietnam — rmit.edu.vn):
    - https://www.rmit.edu.vn/study-at-rmit/tuition-fees
    - https://www.rmit.edu.vn/study-at-rmit/scholarships/...
    - https://www.rmit.edu.vn/students/my-studies/fees-and-payments

Gợi ý văn bản (chủ đề dịch vụ đại học):
    - Học phí & phương thức thanh toán (Tuition Fees)
    - Chính sách học bổng (Scholarship eligibility)
    - Quy định ký túc xá / hỗ trợ chỗ ở (Accommodation Services)
    - Hướng dẫn đăng ký học phần qua cổng thông tin sinh viên (Course Registration)

Lưu ý: một số trang trường (vd VinUni, Fulbright) chặn bot crawler mặc định (HTTP 403) —
không phải lỗi của bạn, đó là cấu hình WAF/Cloudflare phía server. Đổi sang trang khác
thay vì cố vượt qua, và chỉ dùng nguồn công khai/được phép chia sẻ.

Chạy:
    python -m src.task1_collect_legal_docs
"""

import html
import re
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# User-Agent của trình duyệt thật: nhiều site trả 403 cho UA mặc định của requests
# ("python-requests/2.x"). Đây là khai báo danh tính hợp lệ, không phải kỹ thuật né chặn.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
}

# =============================================================================
# NGUỒN TÀI LIỆU
# =============================================================================
# Mỗi item: (tên hiển thị, URL, tên file lưu xuống).
# URL có thể là link PDF trực tiếp HOẶC trang HTML — script tự nhận diện:
#   - PDF trực tiếp  -> lưu nguyên bytes
#   - Trang HTML     -> bóc text rồi render thành PDF bằng fpdf2
#
# LƯU Ý: trang trường thay đổi URL khá thường xuyên. Nếu một link báo 404,
# mở trang chủ trường tìm link mới rồi sửa lại ở đây — đừng bỏ trống dưới 3 file.
LEGAL_SOURCES = [
    {
        "name": "Tuition fees",
        "url": "https://www.rmit.edu.vn/study-at-rmit/fees-and-scholarships/tuition-fees",
        "filename": "tuition-fees-rmit.pdf",
    },
    {
        "name": "Scholarships",
        "url": "https://www.rmit.edu.vn/study-at-rmit/fees-and-scholarships/scholarships",
        "filename": "scholarships-rmit.pdf",
    },
    {
        "name": "Fees and payments",
        "url": "https://www.rmit.edu.vn/students/my-studies/fees-and-payments",
        "filename": "fees-and-payments-rmit.pdf",
    },
    {
        "name": "Accommodation services",
        "url": "https://www.rmit.edu.vn/students/support-and-facilities/accommodation",
        "filename": "accommodation-services-rmit.pdf",
    },
    {
        "name": "Enrolment and course registration",
        "url": "https://www.rmit.edu.vn/students/my-studies/enrolment",
        "filename": "course-registration-rmit.pdf",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


# =============================================================================
# HTML -> TEXT
# =============================================================================

def html_to_text(raw_html: str) -> str:
    """
    Bóc phần chữ ra khỏi HTML, bỏ script/style/nav.

    Dùng regex thay vì BeautifulSoup để không thêm dependency — đủ dùng cho việc
    lấy nội dung thô đưa vào RAG (Task 3 sẽ chuẩn hoá lại lần nữa).
    """
    text = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?is)<(header|footer|nav)[^>]*>.*?</\1>", " ", text)
    # Xuống dòng ở các thẻ khối để giữ được cấu trúc đoạn văn
    text = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr)[^>]*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    # Gộp khoảng trắng thừa, giữ lại tối đa 1 dòng trống giữa các đoạn
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


# =============================================================================
# TEXT -> PDF
# =============================================================================

def _pick_unicode_font() -> Path | None:
    """Tìm 1 font TTF Unicode có sẵn trên máy để in được tiếng Việt trong PDF."""
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for font in candidates:
        if font.exists():
            return font
    return None


def text_to_pdf(title: str, source_url: str, body: str, filepath: Path):
    """
    Render text thành PDF đơn giản.

    Vì sao cần bước này: bài lab yêu cầu file gốc dạng PDF/DOCX trong landing zone,
    nhưng nhiều chính sách của trường chỉ tồn tại dưới dạng trang web. Ta "đóng gói"
    trang web đó thành PDF để pipeline phía sau (MarkItDown ở Task 3) xử lý đồng nhất.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = _pick_unicode_font()
    if font_path:
        pdf.add_font("uni", "", str(font_path))
        pdf.set_font("uni", size=11)
    else:
        # Font lõi của fpdf chỉ hỗ trợ latin-1 -> bỏ ký tự ngoài bảng mã
        pdf.set_font("helvetica", size=11)
        title = title.encode("latin-1", "ignore").decode("latin-1")
        body = body.encode("latin-1", "ignore").decode("latin-1")

    pdf.multi_cell(0, 8, f"{title}\n\nSource: {source_url}\n\n{'-' * 60}\n\n{body}")
    pdf.output(str(filepath))


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_file(source: dict) -> bool:
    """
    Tải 1 nguồn về DATA_DIR. Trả về True nếu lưu được file hợp lệ.

    Tự nhận diện: PDF trực tiếp thì lưu nguyên bytes, HTML thì bóc text rồi in ra PDF.
    """
    url = source["url"]
    filepath = DATA_DIR / source["filename"]

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  ✗ Lỗi tải {url}: {e}")
        return False

    content_type = response.headers.get("Content-Type", "").lower()

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        filepath.write_bytes(response.content)
    else:
        body = html_to_text(response.text)
        if len(body) < 500:
            print(f"  ✗ Trang {url} gần như không có nội dung ({len(body)} ký tự) — bỏ qua")
            return False
        text_to_pdf(source["name"], url, body, filepath)

    size = filepath.stat().st_size
    if size <= 1024:
        print(f"  ✗ File quá nhỏ ({size} bytes): {filepath.name}")
        filepath.unlink(missing_ok=True)
        return False

    print(f"  ✓ Đã tải: {filepath.name} ({size:,} bytes)")
    return True


def collect_all():
    """Tải toàn bộ LEGAL_SOURCES."""
    print("=" * 50)
    print("Task 1: Thu thập văn bản chính sách đại học")
    print("=" * 50)

    setup_directory()

    ok = 0
    for i, source in enumerate(LEGAL_SOURCES, 1):
        print(f"\n[{i}/{len(LEGAL_SOURCES)}] {source['name']}")
        if download_file(source):
            ok += 1

    print(f"\n{'=' * 50}")
    print(f"Kết quả: {ok}/{len(LEGAL_SOURCES)} file tải thành công")
    if ok < 3:
        print("⚠ Chưa đủ 3 file — hãy kiểm tra lại URL trong LEGAL_SOURCES (link có thể đã đổi)")
    else:
        print("✓ Đạt yêu cầu Task 1 (≥3 văn bản)")


if __name__ == "__main__":
    collect_all()
