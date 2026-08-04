"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
(run_pipeline() bên dưới đã tự xoá collection cũ trước khi index lại.)

Chạy:
    python -m src.task4_chunking_indexing
"""

import os
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chunking: RecursiveCharacterTextSplitter.
# Vì sao: nó cắt theo thứ tự ưu tiên đoạn văn -> dòng -> câu -> từ, nên hạn chế
# cắt ngang giữa câu; đồng thời đảm bảo mọi chunk <= CHUNK_SIZE (MarkdownHeader
# splitter không đảm bảo điều này, một section dài sẽ thành 1 chunk khổng lồ).
CHUNK_SIZE = 800        # ~200 token: đủ trọn 1-2 đoạn chính sách, chưa loãng ngữ nghĩa
CHUNK_OVERLAP = 100     # 12.5% overlap: câu nằm ở ranh giới vẫn xuất hiện đủ ngữ cảnh ở 1 chunk
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding: BAAI/bge-m3.
# Vì sao: tài liệu là tiếng Anh (trang trường) nhưng câu hỏi của sinh viên là tiếng Việt.
# bge-m3 là model đa ngữ, embed 2 ngôn ngữ vào cùng không gian vector nên hỏi tiếng Việt
# vẫn match được đoạn tiếng Anh. Model nhẹ hơn (all-MiniLM-L6-v2) chỉ mạnh với tiếng Anh.
# Máy yếu / mạng chậm: đổi sang "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# (~470MB, 384 dim) — nhớ sửa EMBEDDING_DIM cho khớp và xoá chroma_db/ để index lại.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

# Vector store: ChromaDB — chạy local, persist ra thư mục, không cần Docker.
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"


# =============================================================================
# SHARED RESOURCES — Task 5 và 6 import lại từ đây để dùng chung
# =============================================================================

_embedding_model = None
_chroma_client = None


def get_embedding_model():
    """
    Trả về SentenceTransformer đã cache.

    Cache ở cấp module vì load bge-m3 mất vài giây + ~2GB RAM; Task 5 gọi hàm này
    mỗi lần search nên không thể load lại từ đầu.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model: {EMBEDDING_MODEL} (lần đầu sẽ tải model về máy)")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_client():
    """Trả về ChromaDB PersistentClient đã cache (chỉ mở 1 kết nối tới chroma_db/)."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def get_collection():
    """
    Trả về Chroma collection (tạo mới nếu chưa có).

    metadata={"hnsw:space": "cosine"} bắt buộc phải khai lúc TẠO collection —
    mặc định Chroma dùng L2, khi đó công thức score = 1 - distance ở Task 5 sẽ sai.
    """
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []

    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        # Thư mục cha cho biết đây là văn bản chính sách hay bài viết
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        for i, chunk_text in enumerate(splitter.split_text(doc["content"])):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 50:  # bỏ mẩu vụn (dòng kẻ, tiêu đề cụt) — chỉ gây nhiễu
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return chunks

    model = get_embedding_model()
    texts = [c["content"] for c in chunks]

    # normalize_embeddings=True: vector về độ dài 1 -> cosine distance của Chroma
    # đúng bằng 1 - cosine similarity, nhờ đó score ở Task 5 nằm gọn trong [0,1].
    embeddings = model.encode(
        texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if not chunks:
        print("⚠ Không có chunk nào để index")
        return

    collection = get_collection()

    ids = [
        f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}"
        for c in chunks
    ]

    # upsert thay vì add: chạy lại script không bị lỗi trùng ID
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def reset_collection():
    """Xoá collection cũ trước khi index lại, tránh lẫn chunk của corpus cũ."""
    try:
        get_client().delete_collection(COLLECTION_NAME)
        print(f"✓ Đã xoá collection cũ: {COLLECTION_NAME}")
    except Exception:
        pass  # chưa từng tồn tại — bình thường ở lần chạy đầu


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")
    if not docs:
        print("⚠ data/standardized/ chưa có file .md — chạy Task 1-3 trước")
        return

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    reset_collection()
    index_to_vectorstore(chunks)
    print(f"✓ Indexed to vector store ({get_collection().count()} chunks trong collection)")


if __name__ == "__main__":
    run_pipeline()
