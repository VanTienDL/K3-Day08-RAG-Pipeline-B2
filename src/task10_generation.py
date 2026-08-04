"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.

Chạy:
    python -m src.task10_generation
"""

import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# LLM model (OpenRouter model ID). Đổi qua .env nếu tài khoản chưa có credit:
# LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

NO_EVIDENCE_ANSWER = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

# Câu trả lời "không đủ bằng chứng" được ghép từ NO_EVIDENCE_ANSWER để chuỗi LLM sinh
# ra khớp CHÍNH XÁC chuỗi mà code trả về ở nhánh không có kết quả — Task 5/6 (eval)
# đối chiếu bằng so khớp chuỗi, lệch một dấu chấm là tính thành 2 câu trả lời khác nhau.
SYSTEM_PROMPT = f"""Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Tuition Fees, 2026]
3. Nếu context không đủ thông tin → trả lời đúng câu: "{NO_EVIDENCE_ANSWER}"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if not chunks:
        return []

    # Tạo list mới để không làm thay đổi thứ tự kết quả retrieval ban đầu.
    items = list(chunks)
    front = items[::2]
    back = items[1::2]
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts: list[str] = []
    # Đánh số theo tài liệu THỰC SỰ đưa vào prompt. Dùng chỉ số của vòng lặp thì chunk
    # rỗng bị bỏ qua sẽ để lại lỗ hổng ("Tài liệu 1", "Tài liệu 3") và LLM trích dẫn
    # theo số thứ tự không khớp với danh sách nguồn hiển thị cho người dùng.
    i = 0
    for chunk in chunks:
        content = str(chunk.get("content") or "").strip()
        if not content:
            continue

        i += 1
        metadata = chunk.get("metadata") or {}
        source = _first_value(
            metadata, "source", "title", "filename", "file_name", default=f"Nguồn {i}"
        )
        title = _first_value(metadata, "title", default=source)
        year = _first_value(metadata, "year", "published_year", "date", default="không rõ năm")
        doc_type = _first_value(metadata, "type", "doc_type", default="không rõ")
        url = _first_value(metadata, "url", "source_url", default="Không có")

        context_parts.append(
            f"[Tài liệu {i}]\n"
            f"Nguồn: {source}\n"
            f"Tiêu đề: {title}\n"
            f"Năm: {year}\n"
            f"Loại: {doc_type}\n"
            f"URL: {url}\n"
            f"Nội dung:\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)


def _first_value(metadata: dict[str, Any], *keys: str, default: str) -> str:
    """Lấy metadata đầu tiên có giá trị và chuẩn hoá thành chuỗi."""
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _create_llm_client():
    """
    Tạo OpenAI-compatible client cho OpenRouter hoặc OpenAI.

    Trả (None, None) khi không có key HOẶC chưa cài SDK — nơi gọi sẽ quay về
    _extractive_answer(). Không có key thì không được import openai: máy chưa cài
    SDK sẽ ném ImportError và giết luôn nhánh dự phòng.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not openrouter_key and not openai_key:
        return None, None

    try:
        from openai import OpenAI
    except ImportError:
        print("  ⚠ Chưa cài SDK: pip install openai — dùng câu trả lời trích dẫn nguyên văn")
        return None, None

    if openrouter_key:
        return OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=60.0,
        ), LLM_MODEL
    if openai_key:
        # OpenRouter IDs có dạng provider/model; OpenAI API cần tên model thuần.
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAI(api_key=openai_key, timeout=60.0), model

    return None, None


# =============================================================================
# GENERATION
# =============================================================================

def _extractive_answer(chunks: list[dict]) -> str:
    """
    Câu trả lời dự phòng khi chưa cấu hình API key LLM.

    Không sinh chữ mới — chỉ trích nguyên văn đoạn liên quan nhất kèm nguồn, để
    demo/pytest vẫn chạy được. Có nhãn cảnh báo để không nhầm là output của LLM.
    """
    lines = ["⚠ Chưa thể dùng LLM — dưới đây là trích dẫn nguyên văn từ tài liệu:"]
    for index, chunk in enumerate(chunks[:3], 1):
        metadata = chunk.get("metadata") or {}
        source = _first_value(
            metadata, "source", "title", "filename", default=f"Nguồn {index}"
        )
        year = _first_value(
            metadata, "year", "published_year", "date", default="không rõ năm"
        )
        content = str(chunk.get("content") or "").strip()[:400]
        if content:
            lines.append(f"- {content} [{source}, {year}]")
    return "\n".join(lines) if len(lines) > 1 else NO_EVIDENCE_ANSWER


def generate_with_citation(query: str, top_k: int = TOP_K, use_reranking: bool = True) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        top_k: Số chunks đưa vào context

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    query = str(query or "").strip()
    if not query:
        raise ValueError("Câu hỏi không được để trống.")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k phải là số nguyên lớn hơn 0.")

    try:
        chunks = retrieve(query, top_k=top_k, use_reranking=use_reranking)
    except Exception as exc:
        print(f"  ⚠ Retrieval lỗi: {type(exc).__name__}: {exc}")
        chunks = []

    if not chunks:
        return {
            "answer": NO_EVIDENCE_ANSWER,
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    retrieval_source = chunks[0].get("source", "unknown")
    if not context:
        return {
            "answer": NO_EVIDENCE_ANSWER,
            "sources": chunks,
            "retrieval_source": retrieval_source,
        }

    user_message = (
        "Dưới đây là các tài liệu duy nhất bạn được phép sử dụng. "
        "Hãy trích dẫn theo đúng trường Nguồn và Năm của từng tài liệu.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"CÂU HỎI:\n{query}"
    )
    try:
        client, model = _create_llm_client()
    except Exception as exc:
        # Cấu hình hỏng (base_url sai, biến môi trường proxy lỗi...) không được phép
        # làm hỏng cả câu trả lời — vẫn còn nhánh trích dẫn nguyên văn.
        print(f"  ⚠ Không tạo được LLM client: {type(exc).__name__}: {exc}")
        client, model = None, None

    if client is None:
        answer = _extractive_answer(chunks)
    else:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"  ⚠ LLM lỗi: {type(exc).__name__}: {exc}")
            answer = _extractive_answer(chunks)

    return {
        "answer": answer or NO_EVIDENCE_ANSWER,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    test_queries = [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên quốc tế có những học bổng nào?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
