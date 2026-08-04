"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.

Chạy:
    python -m src.task7_reranking
"""

import math
import os

from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
RRF_K = 60  # hằng số làm mượt, lấy từ paper Cormack et al. 2009


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Khác biệt so với bi-encoder (embedding ở Task 4): cross-encoder đọc CẶP
    (query, document) cùng lúc nên chấm điểm liên quan chính xác hơn nhiều,
    đổi lại chậm hơn — vì vậy chỉ dùng cho ~20 candidate đã lọc, không dùng để
    quét toàn corpus.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []
    if not JINA_API_KEY:
        raise RuntimeError("Chưa có JINA_API_KEY trong .env")

    import requests

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {JINA_API_KEY}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
        timeout=30,
    )
    response.raise_for_status()

    reranked = response.json()["results"]
    return [
        {**candidates[r["index"]], "score": float(r["relevance_score"])}
        for r in reranked
    ]


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity giữa 2 vector (thuần Python, không cần numpy)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    selected: list[int] = []
    results: list[dict] = []
    remaining = list(range(len(candidates)))

    while remaining and len(results) < top_k:
        best_idx, best_score = None, float("-inf")

        for idx in remaining:
            emb = candidates[idx].get("embedding")
            if emb is None:
                continue

            relevance = cosine_sim(query_embedding, emb)

            # Độ giống nhất với những cái ĐÃ chọn — càng giống càng bị trừ điểm,
            # nhờ đó tránh 5 kết quả cùng nói một ý (thường gặp do chunk overlap).
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sel_emb = candidates[sel_idx].get("embedding")
                if sel_emb is not None:
                    max_sim_to_selected = max(max_sim_to_selected, cosine_sim(emb, sel_emb))

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score, best_idx = mmr_score, idx

        if best_idx is None:  # không candidate nào có embedding
            break

        item = dict(candidates[best_idx])
        item["score"] = round(best_score, 4)
        results.append(item)

        selected.append(best_idx)
        remaining.remove(best_idx)

    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = RRF_K
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Vì sao hợp để gộp semantic + BM25: hai bên cho điểm ở 2 thang hoàn toàn khác
    nhau (cosine 0-1 vs BM25 0-30+), cộng thẳng thì BM25 lấn át. RRF chỉ dùng
    THỨ HẠNG nên không cần chuẩn hoá thang điểm.

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (k + rank)
            # Giữ bản ghi đầu tiên gặp được (đã có metadata đầy đủ)
            content_map.setdefault(key, item)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = dict(content_map[content])
        # Giữ điểm retrieval GỐC (cosine hoặc BM25) để debug/hiển thị.
        # setdefault: Task 9 chạy RRF 2 lần (merge rồi rerank) — nếu gán đè thì lần 2
        # sẽ ghi điểm RRF của lần 1 vào đây và mất luôn điểm gốc.
        item.setdefault("orig_score", item.get("score"))
        item["score"] = round(score, 6)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        try:
            return rerank_cross_encoder(query, candidates, top_k)
        except Exception as e:
            print(f"  ⚠ Cross-encoder lỗi ({type(e).__name__}: {e}) — chuyển sang RRF")
            return rerank_rrf([candidates], top_k=top_k)

    if method == "mmr":
        try:
            candidates = _attach_embeddings(candidates)
            query_embedding = _embed(query)
            return rerank_mmr(query_embedding, candidates, top_k)
        except Exception as e:
            print(f"  ⚠ MMR lỗi ({type(e).__name__}: {e}) — chuyển sang RRF")
            return rerank_rrf([candidates], top_k=top_k)

    if method == "rrf":
        # Chỉ có 1 danh sách -> RRF giữ nguyên thứ tự và gán lại điểm theo thứ hạng.
        # Muốn gộp nhiều nguồn (Task 9), gọi thẳng rerank_rrf([dense, sparse]).
        return rerank_rrf([candidates], top_k=top_k)

    raise ValueError(f"Unknown rerank method: {method}")


def _embed(text: str) -> list[float]:
    """Embed 1 chuỗi bằng đúng model của Task 4 (dùng cho MMR)."""
    try:
        from .task4_chunking_indexing import get_embedding_model
    except ImportError:
        from src.task4_chunking_indexing import get_embedding_model

    return get_embedding_model().encode(text, normalize_embeddings=True).tolist()


def _attach_embeddings(candidates: list[dict]) -> list[dict]:
    """Bổ sung 'embedding' cho candidate nào chưa có (MMR bắt buộc cần vector)."""
    missing = [c for c in candidates if "embedding" not in c]
    if missing:
        try:
            from .task4_chunking_indexing import get_embedding_model
        except ImportError:
            from src.task4_chunking_indexing import get_embedding_model

        model = get_embedding_model()
        vectors = model.encode(
            [c["content"] for c in missing], normalize_embeddings=True
        )
        for c, vec in zip(missing, vectors):
            c["embedding"] = vec.tolist()
    return candidates


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {}},
    ]
    results = rerank("tuition fee payment", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
