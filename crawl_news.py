import asyncio
from pathlib import Path
from crawl4ai import AsyncWebCrawler

URLS = [
    "https://fami.hust.edu.vn/tin_tuc_dao_tao/hoc-bong-sau-dai-hoc-ap-dung-tu-nam-hoc-2022-2023/",
    "https://fami.hust.edu.vn/thong-bao/danh-gia-cac-hoc-phan-toan-cao-cap-ap-dung-tu-nam-hoc-2023-2024/",
    "https://fami.hust.edu.vn/tin_tuc_dao_tao/gioi-thieu-chuyen-nganh-dao-tao-he-thong-thong-tin-quan-ly/",
    "https://fami.hust.edu.vn/thong-bao/thay-doi-dia-diem-van-phong-khoa-toan-tin/",
    "https://fami.hust.edu.vn/thong-bao/lich-kiem-tra-danh-gia-lien-tuc-cac-hoc-phan-toan-cao-cap-hoc-ky-1-nam-hoc-2024-2025/"
]

SAVE_DIR = Path("data/landing/news")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

async def main():
    async with AsyncWebCrawler() as crawler:
        for i, url in enumerate(URLS, start=1):
            result = await crawler.arun(url)

            with open(
                SAVE_DIR / f"article_{i}.html",
                "w",
                encoding="utf-8"
            ) as f:
                f.write(result.cleaned_html)

            print(f"Saved article_{i}.html")

asyncio.run(main())