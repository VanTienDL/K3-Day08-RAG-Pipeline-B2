"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.

Chạy:
    python -m src.task8_pageindex_vectorless
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Lưu doc_id sau khi upload để lần sau không phải upload lại (tốn quota + thời gian)
DOC_IDS_FILE = Path(__file__).parent.parent / "data" / "pageindex_docs.json"

DEBUG = os.getenv("PAGEINDEX_DEBUG", "").lower() in ("1", "true")


def _get_client():
    """Khởi tạo PageIndexClient. Trả về None nếu thiếu key hoặc chưa cài SDK."""
    if not PAGEINDEX_API_KEY:
        print("⚠ Chưa có PAGEINDEX_API_KEY trong .env — bỏ qua PageIndex")
        return None
    try:
        from pageindex.client import PageIndexClient
    except ImportError:
        print("⚠ Chưa cài SDK: pip install pageindex")
        return None
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _markdown_to_pdf(md_file: Path, out_dir: Path) -> Path:
    """
    Đổi 1 file .md sang PDF vì PageIndex nhận PDF chứ không nhận markdown.

    Dùng lại hàm render PDF của Task 1 để không lặp code font Unicode.
    """
    try:
        from .task1_collect_legal_docs import text_to_pdf
    except ImportError:
        from src.task1_collect_legal_docs import text_to_pdf

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{md_file.stem}.pdf"
    text_to_pdf(md_file.stem, md_file.name, md_file.read_text(encoding="utf-8"), pdf_path)
    return pdf_path


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.

    Ghi lại {tên file: doc_id} vào data/pageindex_docs.json để pageindex_search() dùng.
    """
    client = _get_client()
    if client is None:
        return {}

    tmp_dir = Path(__file__).parent.parent / "data" / "_pageindex_pdf"
    doc_ids = {}

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        try:
            pdf_path = _markdown_to_pdf(md_file, tmp_dir)
            resp = client.submit_document(str(pdf_path))
            if DEBUG:
                print(json.dumps(resp, indent=2, ensure_ascii=False)[:800])

            doc_id = resp.get("doc_id") or resp.get("id")
            if doc_id:
                doc_ids[md_file.name] = doc_id
                print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")
            else:
                print(f"  ✗ Không lấy được doc_id cho {md_file.name}: {resp}")
        except Exception as e:
            print(f"  ✗ Lỗi upload {md_file.name}: {type(e).__name__}: {e}")

    if doc_ids:
        DOC_IDS_FILE.write_text(json.dumps(doc_ids, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✓ Đã lưu {len(doc_ids)} doc_id vào {DOC_IDS_FILE.name}")

    return doc_ids


def _load_doc_ids() -> dict:
    """Đọc danh sách doc_id đã upload."""
    if DOC_IDS_FILE.exists():
        try:
            return json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _wait_for_retrieval(client, retrieval_id: str, timeout: int = 60) -> dict:
    """Poll cho đến khi truy vấn xong (PageIndex xử lý bất đồng bộ)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        retrieval = client.get_retrieval(retrieval_id)
        status = retrieval.get("status", "")
        if status in ("completed", "success", "done"):
            return retrieval
        if status in ("failed", "error"):
            raise RuntimeError(f"PageIndex retrieval thất bại: {retrieval}")
        time.sleep(2)
    raise TimeoutError("PageIndex retrieval quá thời gian chờ")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
        Trả về [] nếu chưa cấu hình PageIndex — Task 9 sẽ tự dùng kết quả hybrid.
    """
    client = _get_client()
    if client is None:
        return []

    doc_ids = _load_doc_ids()
    if not doc_ids:
        print("⚠ Chưa upload document nào lên PageIndex — chạy upload_documents() trước")
        return []

    results: list[dict] = []

    for filename, doc_id in doc_ids.items():
        if len(results) >= top_k:
            break
        try:
            resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")
            retrieval = _wait_for_retrieval(client, retrieval_id)

            if DEBUG:
                print(json.dumps(retrieval, indent=2, ensure_ascii=False)[:1500])

            # PageIndex không trả score -> tự gán điểm giảm dần theo thứ hạng,
            # để Task 9/10 vẫn sắp xếp và hiển thị được như các nguồn khác.
            for node in retrieval.get("retrieved_nodes", []):
                for group in node.get("relevant_contents", []):
                    for item in group:
                        content = (item.get("relevant_content") or "").strip()
                        if not content:
                            continue
                        results.append({
                            "content": content,
                            "score": round(1.0 / (len(results) + 1), 4),
                            "metadata": {
                                "source": filename,
                                "type": "pageindex",
                                "section": item.get("section_title", ""),
                            },
                            "source": "pageindex",
                        })
        except Exception as e:
            print(f"  ✗ PageIndex lỗi với {filename}: {type(e).__name__}: {e}")

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
