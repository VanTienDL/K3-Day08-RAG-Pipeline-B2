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
import sys

# Windows UTF-8 fix
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

from dotenv import load_dotenv

# Bắt buộc: embedding gọi API nên cần OPENAI_API_KEY. Chạy `python -m src.task4_...`
# trực tiếp mà không load .env thì key sẽ không thấy được.
load_dotenv()

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

# Embedding: OpenAI text-embedding-3-small (gọi qua API, KHÔNG chạy model tại chỗ).
# Vì sao: tài liệu là tiếng Anh (trang trường) nhưng câu hỏi của sinh viên là tiếng Việt.
# Model này đa ngữ nên hỏi tiếng Việt vẫn match được đoạn tiếng Anh — tương đương
# BAAI/bge-m3 về mặt này, nhưng không phải cài torch (~1GB) và tải model (~2.2GB).
# Đánh đổi: mỗi lần search đều cần mạng, và cần OPENAI_API_KEY thật (key OpenRouter
# KHÔNG dùng được — OpenRouter không có endpoint embeddings).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536  # text-embedding-3-large là 3072 — đổi model nhớ sửa số này

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
    Trả về OpenAI client đã cache (dùng cho endpoint embeddings).

    Cache ở cấp module vì Task 5 gọi mỗi lần search — tạo client mới liên tục sẽ
    mở lại kết nối HTTP không cần thiết.

    Lưu ý: PHẢI là key OpenAI thật (sk-...). OpenRouter chỉ phục vụ chat completions,
    không có endpoint /v1/embeddings, nên OPENROUTER_API_KEY không dùng được ở đây.
    """
    global _embedding_model
    if _embedding_model is None:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Thiếu OPENAI_API_KEY trong .env — embedding không chạy được. "
                "Key OpenRouter KHÔNG thay thế được vì OpenRouter không có endpoint embeddings."
            )
        _embedding_model = OpenAI(api_key=api_key, timeout=60.0)
    return _embedding_model


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Embed nhiều đoạn văn bằng 1 loạt request tới OpenAI.

    Gửi theo lô thay vì từng đoạn một: 300 chunk mà gọi 300 request thì vừa chậm
    vừa dễ dính rate limit, gộp thành 3 request thì xong trong vài giây.

    Vector OpenAI trả về đã chuẩn hoá sẵn về độ dài 1, nên cosine distance của Chroma
    đúng bằng 1 - cosine similarity — công thức score ở Task 5 giữ nguyên, không cần
    normalize thủ công như hồi dùng sentence-transformers.
    """
    if not texts:
        return []

    client = get_embedding_model()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        # API không đảm bảo thứ tự trả về -> sắp lại theo index cho chắc
        for item in sorted(response.data, key=lambda d: d.index):
            vectors.append(item.embedding)
        print(f"  ... embedded {min(start + batch_size, len(texts))}/{len(texts)}")

    return vectors


def embed_query(text: str) -> list[float]:
    """Embed 1 câu truy vấn. Dùng chung model với lúc index để vector cùng không gian."""
    return embed_texts([text])[0]


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

        # Thư mục con NGAY DƯỚI standardized/ cho biết đây là văn bản chính sách hay
        # bài viết. Chỉ xét phần đường dẫn tương đối — nếu xét md_file.parts (đường dẫn
        # tuyệt đối) thì một thư mục cha bất kỳ tên "legal" trên máy sẽ gán nhầm type.
        relative = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = "legal" if relative.parts[0] == "legal" else "news"
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

    embeddings = embed_texts([c["content"] for c in chunks])

    # Đổi EMBEDDING_MODEL mà quên sửa EMBEDDING_DIM là lỗi hay gặp: Chroma sẽ nhận
    # vector sai chiều và báo lỗi khó hiểu ở tận bước index. Chặn ngay tại đây.
    actual_dim = len(embeddings[0])
    if actual_dim != EMBEDDING_DIM:
        raise ValueError(
            f"EMBEDDING_DIM={EMBEDDING_DIM} không khớp model {EMBEDDING_MODEL} "
            f"(thật sự là {actual_dim}). Sửa EMBEDDING_DIM và xoá chroma_db/ để index lại."
        )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if not chunks:
        print("⚠ Không có chunk nào để index")
        return

    collection = get_collection()

    # Có type trong ID: legal/ và news/ có thể chứa 2 file TRÙNG TÊN, khi đó ID trùng
    # nhau và upsert sẽ ghi đè chunk của file này bằng chunk của file kia (mất dữ liệu
    # âm thầm, không báo lỗi).
    ids = [
        f"{c['metadata']['type']}_{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}"
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
