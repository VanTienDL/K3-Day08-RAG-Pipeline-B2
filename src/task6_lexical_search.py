"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Chạy:
    python -m src.task6_lexical_search
"""

import re

try:
    from .task4_chunking_indexing import chunk_documents, get_collection, load_documents
except ImportError:  # khi chạy trực tiếp `python src/task6_lexical_search.py`
    from src.task4_chunking_indexing import chunk_documents, get_collection, load_documents

# Corpus dùng chung cho BM25. Nạp lazy ở lần search đầu tiên (xem _ensure_index).
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

_bm25 = None


def load_corpus() -> list[dict]:
    """
    Nạp corpus cho BM25, ưu tiên lấy từ chính ChromaDB của Task 4.

    Vì sao lấy từ Chroma: Task 9 gộp kết quả dense + sparse bằng RRF, mà RRF khớp
    các kết quả trùng nhau dựa trên chuỗi 'content'. Nếu BM25 tự chunk lại từ file
    .md thì ranh giới chunk sẽ lệch với bên vector store, không chunk nào trùng nhau
    và RRF mất tác dụng gộp.
    """
    try:
        collection = get_collection()
        if collection.count() > 0:
            data = collection.get(include=["documents", "metadatas"])
            return [
                {"content": doc, "metadata": dict(meta or {})}
                for doc, meta in zip(data["documents"], data["metadatas"])
            ]
    except Exception as e:
        print(f"⚠ Không đọc được ChromaDB ({type(e).__name__}) — chunk lại từ file .md")

    # Dự phòng: chưa index thì chunk trực tiếp từ data/standardized/
    try:
        return chunk_documents(load_documents())
    except Exception as e:
        print(f"⚠ Không dựng được corpus dự phòng ({type(e).__name__}: {e}) — "
              "kiểm tra `pip install -r requirements.txt` và chạy Task 3-4 trước")
        return []


def tokenize(text: str) -> list[str]:
    """
    Tách từ đơn giản: lowercase + giữ lại chữ/số (Unicode nên tiếng Việt vẫn đúng).

    BM25 so khớp theo token nên bước này quyết định chất lượng: nếu không lowercase
    thì "Tuition" và "tuition" bị coi là 2 từ khác nhau.
    """
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not corpus:
        return None

    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)  # k1=1.5, b=0.75 mặc định


def _ensure_index():
    """Nạp corpus + dựng index ở lần gọi đầu tiên, các lần sau dùng lại."""
    global CORPUS, _bm25, _tfidf_vectorizer, _tfidf_matrix
    if _bm25 is None:
        CORPUS = load_corpus()
        _bm25 = build_bm25_index(CORPUS)
        # CORPUS vừa đổi -> ma trận TF-IDF cũ (nếu có) đánh số dòng theo corpus cũ,
        # dùng tiếp sẽ trả về nhầm chunk. Xoá cache để nó dựng lại.
        _tfidf_vectorizer = None
        _tfidf_matrix = None
        print(f"✓ BM25 index: {len(CORPUS)} chunks")
    return _bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    bm25 = _ensure_index()
    if bm25 is None:
        return []

    scores = bm25.get_scores(tokenize(query))

    # Sắp xếp giảm dần theo điểm, chỉ giữ chunk thật sự có từ khớp (score > 0)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked[:top_k]:
        if scores[idx] <= 0:
            break
        results.append({
            "content": CORPUS[idx]["content"],
            "score": round(float(scores[idx]), 4),
            # copy: Task 9/10 có thể thêm key vào result, không được đụng vào CORPUS gốc
            "metadata": dict(CORPUS[idx]["metadata"]),
        })

    return results


# =============================================================================
# BONUS — TF-IDF (phương pháp lexical thứ 2, để so sánh trong demo)
# =============================================================================

_tfidf_vectorizer = None
_tfidf_matrix = None


def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Lexical search bằng TF-IDF + cosine similarity (scikit-learn).

    Khác BM25 ở đâu:
        - TF-IDF: trọng số của 1 từ tăng TUYẾN TÍNH theo số lần xuất hiện.
        - BM25: có hệ số bão hoà k1 — từ xuất hiện lần thứ 10 gần như không cộng
          thêm điểm so với lần thứ 9, và có hệ số b phạt document dài.
    => Với đoạn văn dài ngắn lệch nhau nhiều (chính sách vs thông báo), BM25 thường
       công bằng hơn; TF-IDF dễ ưu ái đoạn lặp từ khoá nhiều lần.
    """
    global _tfidf_vectorizer, _tfidf_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer

    _ensure_index()
    if not CORPUS:
        return []

    if _tfidf_matrix is None:
        _tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=True, token_pattern=None)
        _tfidf_matrix = _tfidf_vectorizer.fit_transform([c["content"] for c in CORPUS])

    query_vec = _tfidf_vectorizer.transform([query])
    scores = (_tfidf_matrix @ query_vec.T).toarray().ravel()

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked[:top_k]:
        if scores[idx] <= 0:
            break
        results.append({
            "content": CORPUS[idx]["content"],
            "score": round(float(scores[idx]), 4),
            "metadata": dict(CORPUS[idx]["metadata"]),
        })

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
