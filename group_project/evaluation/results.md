# RAG Evaluation Results

## Framework sử dụng

> Framework đã chọn: **RAGAS** (v0.4.3)

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ (A - B) |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.8521 | 0.8609 | -0.0088 |
| Answer Relevance | 0.6964 | 0.7070 | -0.0106 |
| Context Recall | 1.0000 | 1.0000 | 0.0000 |
| Context Precision | 0.8632 | 0.9062 | -0.0430 |
| **Average** | 0.8529 | 0.8685 | -0.0156 |

---

## A/B Comparison Analysis

**Config A:**
> Hybrid Search (Semantic/ChromaDB + Lexical/BM25) kết hợp Reciprocal Rank Fusion (RRF) và được đánh giá lại (Reranking) bằng CrossEncoder (BGE-Reranker).

**Config B:**
> Dense-only Search (Chỉ sử dụng Semantic Search với ChromaDB và embedding model text-embedding-3-small), không có bước reranking.

**Kết luận:**
> Config B (dense-only) cho kết quả nhỉnh hơn một chút trên tập golden dataset này (Average 0.8685 so với 0.8529). Điều này cho thấy thuật toán Semantic Search đã trích xuất rất tốt các context liên quan (Context Recall đều đạt 1.0). Việc thêm BM25 và Reranker trong Config A có thể đã làm nhiễu một số kết quả hoặc đẩy các chunk không chứa câu trả lời trực tiếp lên trên, làm giảm nhẹ Context Precision (-0.043).

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Học phí học kỳ hè được tính như thế nào so với học phí học kỳ chính? | 1.000 | 0.000 | 1.000 | Evaluation (RAGAS) | **Lỗi đánh giá RAGAS:** RAGAS sinh ngược câu hỏi tiếng Anh từ câu trả lời tiếng Việt, làm Cosine Similarity = 0, dù đáp án đúng. |
| 2 | Bài đánh giá liên tục môn Toán cao cấp được mở theo lịch như thế nào? | 1.000 | 0.935 | 1.000 | Retrieval (Precision 0.20) | **Nhiễu Semantic:** Chunk chứa đáp án đúng bị xếp hạng rất thấp vì từ khóa "Toán cao cấp" có ở quá nhiều chunk khác. |
| 3 | Điều kiện GPA tối thiểu để sinh viên từ học kỳ 2 trở đi được xét Học bổng...? | 0.500 | 0.904 | 1.000 | Generation (Faithfulness) | **Hallucination:** LLM sinh ra câu trả lời gom thêm các điều kiện không liên quan (hoặc không có trong context) làm giảm tính trung thực. |

---

## Recommendations

### Cải tiến 1: Khắc phục lỗi ngôn ngữ khi đánh giá RAGAS
**Action:** Tùy chỉnh (customize) các prompts nội bộ của RAGAS sang tiếng Việt, hoặc ép buộc LLM trả về ngôn ngữ gốc trong quá trình sinh câu hỏi ngược (Reverse Question Generation) của metric Answer Relevancy.
**Expected impact:** Điểm Answer Relevancy sẽ phản ánh đúng chất lượng thật của hệ thống (tăng từ 0.0 lên mức > 0.8), loại bỏ các "false negatives" trong quá trình đánh giá.

### Cải tiến 2: Nâng cấp chiến lược Chunking & Retrieval (để tăng Context Precision)
**Action:** Áp dụng Metadata Filtering (thêm tag loại tài liệu: học phí, học bổng, toán cao cấp) kết hợp Semantic Chunking (chia chunk theo ý nghĩa thay vì cắt cứng số từ). Đồng thời tuning lại trọng số RRF giữa BM25 và Dense Search để BM25 bổ trợ tốt hơn thay vì kéo rank xuống.
**Expected impact:** Cải thiện Context Precision (từ 0.2 lên mức cao hơn) do hệ thống truy xuất được chính xác đoạn văn chứa câu trả lời ngay từ top 1.

### Cải tiến 3: Tinh chỉnh Prompt cho LLM (để tăng Faithfulness)
**Action:** Chỉnh sửa System Prompt của LLM khắt khe hơn: *"Tuyệt đối chỉ sử dụng thông tin trong context cung cấp, không sử dụng kiến thức bên ngoài. Nếu context không nhắc đến, hãy trả lời 'Tôi không tìm thấy thông tin'."*
**Expected impact:** Tăng Faithfulness lên 1.0 đối với các câu hỏi phức tạp bằng cách ngăn chặn LLM tự biên tự diễn (hallucinate) thông tin.
