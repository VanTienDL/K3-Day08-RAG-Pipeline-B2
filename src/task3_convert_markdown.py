"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục

Chạy:
    python -m src.task3_convert_markdown
"""

import json
import re
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def clean_markdown(text: str) -> str:
    """
    Dọn rác sau khi convert: dòng trống thừa, khoảng trắng cuối dòng.

    Bước này quan trọng cho Task 4: chunk sạch thì embedding mới đại diện đúng
    nội dung, tránh chunk toàn ký tự trắng làm nhiễu kết quả retrieval.
    """
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print("  (chưa có data/landing/legal/ — chạy Task 1 trước)")
        return 0

    md = MarkItDown()
    count = 0

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        print(f"Converting: {filepath.name}")
        try:
            result = md.convert(str(filepath))
        except Exception as e:
            print(f"  ✗ Lỗi convert: {type(e).__name__}: {e}")
            continue

        content = clean_markdown(result.text_content or "")
        if len(content) < 200:
            print(f"  ✗ Nội dung sau convert quá ngắn ({len(content)} ký tự) — bỏ qua")
            continue

        # Header giữ lại tên file gốc để Task 10 trích dẫn được nguồn
        header = f"# {filepath.stem.replace('-', ' ').title()}\n\n"
        header += f"**Source file:** {filepath.name}\n**Type:** legal\n\n---\n\n"

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(header + content, encoding="utf-8")
        print(f"  ✓ Saved: {output_path.name} ({len(content):,} ký tự)")
        count += 1

    return count


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print("  (chưa có data/landing/news/ — chạy Task 2 trước)")
        return 0

    count = 0

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"  ✗ File JSON hỏng: {e}")
            continue

        # File crawl4ai đã là markdown sẵn -> chỉ cần gắn metadata header
        header = f"# {data.get('title', 'Unknown')}\n\n"
        header += f"**Source:** {data.get('url', 'N/A')}\n"
        header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n"
        header += "**Type:** news\n\n---\n\n"

        content = clean_markdown(data.get("content_markdown", ""))
        if len(content) < 200:
            print(f"  ✗ Nội dung quá ngắn ({len(content)} ký tự) — bỏ qua")
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(header + content, encoding="utf-8")
        print(f"  ✓ Saved: {output_path.name} ({len(content):,} ký tự)")
        count += 1

    return count


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    n_legal = convert_legal_docs()

    print("\n--- News Articles ---")
    n_news = convert_news_articles()

    print(f"\n✓ Done! {n_legal} legal + {n_news} news -> {OUTPUT_DIR}")
    if n_legal + n_news == 0:
        print("⚠ Không convert được file nào — kiểm tra lại data/landing/")


if __name__ == "__main__":
    convert_all()
