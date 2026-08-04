"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

Chạy:
    python -m src.task5_semantic_search
"""

import os

from dotenv import load_dotenv

# Bắt buộc: HyDE bên dưới đọc OPENROUTER_API_KEY qua os.getenv. Không load .env ở đây
# thì key nằm trong .env sẽ không thấy được, HyDE âm thầm tắt và tưởng là thiếu key.
load_dotenv()

try:
    from .task4_chunking_indexing import embed_query, get_collection
except ImportError:  # khi chạy trực tiếp `python src/task5_semantic_search.py`
    from src.task4_chunking_indexing import embed_query, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    collection = get_collection()

    total = collection.count()
    if total == 0:
        print("⚠ Vector store rỗng — chạy `python -m src.task4_chunking_indexing` trước")
        return []

    # Công thức score = 1 - distance ở dưới CHỈ đúng khi collection dùng cosine.
    # chroma_db/ tạo bằng bản code cũ có thể đang để mặc định L2 (distance không chặn
    # trên bởi 1) -> score bị kẹp về 0 hàng loạt và ngưỡng fallback ở Task 9 sai theo.
    space = (collection.metadata or {}).get("hnsw:space")
    if space != "cosine":
        print(f"⚠ Collection đang dùng space='{space}' thay vì 'cosine' — xoá chroma_db/ "
              "rồi chạy lại `python -m src.task4_chunking_indexing`")

    # Bước 1: embed query bằng ĐÚNG model đã dùng ở Task 4 (vector phải cùng không gian)
    query_vector = embed_query(query)

    # Bước 2: truy vấn Chroma. n_results không được vượt số chunk đang có
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"],
    )

    # Bước 3: đổi cosine distance -> similarity trong [0, 1]
    output = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        score = max(0.0, min(1.0, 1.0 - dist))
        output.append({
            "content": doc,
            "score": round(score, 4),
            "metadata": dict(meta or {}),
        })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


# =============================================================================
# BONUS — HyDE (Hypothetical Document Embeddings)
# =============================================================================

def generate_hypothetical_answer(query: str) -> str:
    """
    Cho LLM viết một câu trả lời giả định cho query.

    Ý tưởng HyDE: câu hỏi ngắn ("học phí bao nhiêu?") và đoạn văn chính sách dài
    có cách diễn đạt rất khác nhau nên vector của chúng không gần nhau lắm. Một câu
    trả lời giả định — dù có thể sai số liệu — lại dùng đúng loại từ ngữ của tài liệu,
    nên embed nó sẽ tìm ra đoạn đúng tốt hơn.

    Trả về "" nếu chưa có API key -> nơi gọi tự quay về search thường.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Write a short factual paragraph (2-3 sentences, in English) that "
                        "would plausibly answer the user's question, as if copied from a "
                        "university policy page. Do not add any preamble."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"  ⚠ HyDE lỗi ({type(e).__name__}) — dùng semantic search thường")
        return ""


def semantic_search_hyde(query: str, top_k: int = 10) -> list[dict]:
    """
    Semantic search dùng HyDE: search bằng "query gốc + câu trả lời giả định".

    Ghép cả 2 (thay vì chỉ dùng câu giả định) để nếu LLM bịa lệch chủ đề thì
    query gốc vẫn kéo kết quả về đúng hướng.
    """
    hypothetical = generate_hypothetical_answer(query)
    if not hypothetical:
        return semantic_search(query, top_k)

    return semantic_search(f"{query}\n\n{hypothetical}", top_k)


if __name__ == "__main__":
    # Test
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
