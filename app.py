"""Streamlit chatbot cho University Services RAG."""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.task10_generation import generate_with_citation  # noqa: E402


st.set_page_config(
    page_title="University Services RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


SUGGESTIONS = [
    "Học phí tại RMIT Vietnam được thanh toán như thế nào?",
    "Điều kiện nhận học bổng Academic Achievement là gì?",
    "Làm thế nào để đặt phòng học nhóm ở thư viện?",
    "Trường có những dịch vụ hỗ trợ chỗ ở nào cho sinh viên?",
    "Sinh viên đăng ký học phần qua myRMIT như thế nào?",
]


def source_label(source: dict, index: int) -> str:
    metadata = source.get("metadata") or {}
    return str(
        metadata.get("title")
        or metadata.get("source")
        or metadata.get("filename")
        or f"Nguồn {index}"
    )


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander(f"📚 Nguồn tham khảo ({len(sources)})"):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata") or {}
            label = source_label(source, index)
            url = metadata.get("url") or metadata.get("source_url")
            score = source.get("score")
            retrieval_source = source.get("source", "không rõ")

            if url:
                st.markdown(f"**{index}. [{label}]({url})**")
            else:
                st.markdown(f"**{index}. {label}**")

            details = [f"retrieval: `{retrieval_source}`"]
            if isinstance(score, (int, float)):
                details.append(f"score: `{score:.4f}`")
            if metadata.get("type"):
                details.append(f"loại: `{metadata['type']}`")
            st.caption(" · ".join(details))

            content = str(source.get("content") or "").strip()
            if content:
                st.write(content[:500] + ("…" if len(content) > 500 else ""))
            if index < len(sources):
                st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


with st.sidebar:
    st.title("🎓 University Services RAG")
    st.caption("Trợ lý hỏi đáp về chính sách và dịch vụ đại học.")
    st.divider()

    st.subheader("⚙️ Thiết lập")
    top_k = st.slider(
        "Số đoạn tài liệu (top_k)",
        min_value=1,
        max_value=10,
        value=5,
        help="Nhiều đoạn hơn có thể tăng độ bao phủ nhưng làm context dài hơn.",
    )
    show_sources = st.toggle("Hiển thị nguồn tham khảo", value=True)
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.divider()
    st.subheader("💡 Câu hỏi gợi ý")
    for index, suggestion in enumerate(SUGGESTIONS):
        if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            st.session_state.pending_query = suggestion

    st.divider()
    st.caption(
        "Hybrid Retrieval → Reranking → PageIndex fallback → "
        "LLM generation có citation"
    )


st.title("🎓 University Services RAG Chatbot")
st.caption(
    "Hỏi về học phí, học bổng, đăng ký học phần, thư viện và các dịch vụ sinh viên."
)

if not st.session_state.messages:
    st.info("Hãy nhập câu hỏi hoặc chọn một câu hỏi gợi ý ở thanh bên.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_sources and message["role"] == "assistant":
            render_sources(message.get("sources", []))


typed_query = st.chat_input("Nhập câu hỏi của bạn…")
query = typed_query or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    query = str(query).strip()
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        answer = ""
        sources: list[dict] = []
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm tài liệu và tổng hợp câu trả lời…"):
                try:
                    result = generate_with_citation(query, top_k=top_k)
                    answer = result.get("answer") or "Không nhận được câu trả lời từ mô hình."
                    sources = result.get("sources") or []
                    st.markdown(answer)
                    if show_sources:
                        render_sources(sources)
                except NotImplementedError as exc:
                    answer = (
                        "⚠️ Pipeline retrieval của Task 9 chưa được hoàn thiện. "
                        "Hãy phối hợp với Role 1 để tích hợp hàm `retrieve()`."
                    )
                    st.warning(answer)
                    st.caption(str(exc))
                except ValueError as exc:
                    answer = f"⚠️ {exc}"
                    st.warning(answer)
                except Exception as exc:
                    answer = "❌ Không thể tạo câu trả lời. Vui lòng kiểm tra cấu hình và thử lại."
                    st.error(answer)
                    # Chi tiết hữu ích khi demo nội bộ nhưng không làm lộ API key.
                    st.caption(f"Chi tiết: {type(exc).__name__}: {exc}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
