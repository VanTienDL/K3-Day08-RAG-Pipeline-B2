"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.

Chạy:
    python -m src.task2_crawl_news
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách bài viết cần crawl (≥5 để đạt yêu cầu Task 2).
# Nếu link nào 404, mở trang chủ trường tìm bài khác thay vào.
ARTICLE_URLS = [
    "https://www.rmit.edu.vn/students/support-and-facilities/library",
    "https://www.rmit.edu.vn/students/support-and-facilities/student-support",
    "https://www.rmit.edu.vn/students/support-and-facilities/health-and-wellbeing",
    "https://www.rmit.edu.vn/students/my-studies/enrolment",
    "https://www.rmit.edu.vn/study-at-rmit/fees-and-scholarships/scholarships",
    "https://www.rmit.edu.vn/students/support-and-facilities/careers-and-employability",
]


def slugify(url: str) -> str:
    """Đổi URL thành tên file an toàn, ví dụ .../support/library -> support-library."""
    path = re.sub(r"^https?://[^/]+/?", "", url).strip("/")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-").lower()
    return (slug or "article")[:60]


# =============================================================================
# CRAWLER
# =============================================================================

async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)

        # crawl4ai bản mới trả markdown là object (có .raw_markdown), bản cũ trả str
        markdown = getattr(result, "markdown", "") or ""
        markdown = getattr(markdown, "raw_markdown", markdown)

        metadata = getattr(result, "metadata", None) or {}
        title = metadata.get("title") or url

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": str(markdown),
        }


def crawl_article_simple(url: str) -> dict:
    """
    Phương án dự phòng khi chưa cài được crawl4ai/playwright.

    Chỉ dùng requests + bóc thẻ HTML, không chạy JavaScript — nội dung sẽ ít hơn
    nhưng vẫn đủ để pipeline chạy thông. Ưu tiên crawl_article() nếu cài được.
    """
    import requests

    from .task1_collect_legal_docs import HEADERS, html_to_text

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", response.text)
    title = match.group(1).strip() if match else url

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": html_to_text(response.text),
    }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    try:
        import crawl4ai  # noqa: F401
        use_crawl4ai = True
    except ImportError:
        print("⚠ Chưa cài crawl4ai — dùng phương án dự phòng requests (nội dung ít hơn)")
        use_crawl4ai = False

    saved = 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            if use_crawl4ai:
                article = await crawl_article(url)
            else:
                article = crawl_article_simple(url)
        except Exception as e:
            print(f"  ✗ Lỗi crawl: {type(e).__name__}: {e}")
            continue

        if len(article["content_markdown"]) < 500:
            print("  ✗ Nội dung quá ngắn, có thể bị chặn — bỏ qua")
            continue

        filename = f"{i:02d}-{slugify(url)}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  ✓ Saved: {filepath.name} ({len(article['content_markdown']):,} ký tự)")
        saved += 1

    print(f"\nKết quả: {saved}/{len(ARTICLE_URLS)} bài crawl thành công")
    if saved < 5:
        print("⚠ Chưa đủ 5 bài — bổ sung thêm URL vào ARTICLE_URLS rồi chạy lại")
    else:
        print("✓ Đạt yêu cầu Task 2 (≥5 bài viết)")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(crawl_all())
