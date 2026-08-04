"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.

Chạy:
    python -m src.task9_retrieval_pipeline
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# ⚠ CHƯA ĐO — 0.35 chỉ là giá trị tạm cho text-embedding-3-small.
# Ngưỡng 0.48 cũ đo trên bge-m3, KHÔNG dùng lại được: mỗi embedding model cho một
# dải điểm cosine khác nhau, model OpenAI thường cho điểm thấp hơn bge-m3 rõ rệt nên
# giữ 0.48 sẽ khiến fallback kích hoạt gần như mọi câu hỏi.
# BẮT BUỘC: sau khi index xong, chạy `python -m src.task9_retrieval_pipeline` để
# calibrate_threshold() in ra 2 vùng điểm, rồi thay số ở đây bằng số nằm giữa.
SCORE_THRESHOLD = 0.35  # Nếu best score (cosine gốc) < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: chạy 2 nhánh retrieval. Lấy dư (top_k * 2) để RRF có đủ nguyên liệu gộp.
    # Bọc try/except để 1 nhánh hỏng (chưa index, thiếu thư viện) không giết cả pipeline.
    try:
        dense_results = semantic_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"  ⚠ Semantic search lỗi: {type(e).__name__}: {e}")
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"  ⚠ Lexical search lỗi: {type(e).__name__}: {e}")
        sparse_results = []

    # Step 2: gộp 2 danh sách bằng RRF (chỉ dựa vào thứ hạng nên không cần chuẩn hoá điểm)
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: rerank lần cuối để chọn ra top_k
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item.setdefault("source", "hybrid")
    else:
        final_results = merged[:top_k]

    # Step 4: quyết định fallback bằng ĐIỂM COSINE GỐC, KHÔNG phải điểm RRF.
    # dense_results đã sort giảm dần nên phần tử [0] là điểm cao nhất.
    best_score = dense_results[0]["score"] if dense_results else 0.0
    if best_score < score_threshold:
        print(f"  ⚠ Semantic best score ({best_score:.3f}) < threshold ({score_threshold})"
              " → thử fallback PageIndex")
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback[:top_k]
        # PageIndex chưa cấu hình / không có kết quả → vẫn trả hybrid để không rỗng

    return final_results[:top_k]


def calibrate_threshold(relevant_queries: list[str], off_topic_queries: list[str]):
    """
    Công cụ tự đo ngưỡng SCORE_THRESHOLD cho corpus của nhóm bạn.

    Cách dùng: đưa vào vài câu chắc chắn có trong tài liệu và vài câu chắc chắn lạc đề,
    rồi chọn một số nằm giữa 2 vùng điểm in ra.
    """
    def best(q: str) -> float:
        try:
            hits = semantic_search(q, top_k=1)
            return hits[0]["score"] if hits else 0.0
        except Exception:
            return 0.0

    print("\n--- Câu hỏi ĐÚNG chủ đề ---")
    relevant_scores = []
    for q in relevant_queries:
        s = best(q)
        relevant_scores.append(s)
        print(f"  {s:.3f}  {q}")

    print("\n--- Câu hỏi LẠC ĐỀ ---")
    off_scores = []
    for q in off_topic_queries:
        s = best(q)
        off_scores.append(s)
        print(f"  {s:.3f}  {q}")

    if relevant_scores and off_scores:
        low = min(relevant_scores)
        high = max(off_scores)
        print(f"\nGợi ý threshold: khoảng {(low + high) / 2:.2f} "
              f"(thấp nhất của nhóm đúng: {low:.3f}, cao nhất của nhóm lạc đề: {high:.3f})")


if __name__ == "__main__":
    test_queries = [
        "What is the tuition fee at RMIT Vietnam?",
        "How do I book a library study room?",
        "What scholarships are available for international students?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")

    print("\n" + "=" * 60)
    print("Calibrate threshold")
    print("=" * 60)
    calibrate_threshold(
        relevant_queries=[
            "tuition fee payment",
            "scholarship eligibility",
            "library opening hours",
        ],
        off_topic_queries=[
            "xyzabc123nonsense",
            "cách nấu phở bò",
            "premier league top scorer",
        ],
    )
