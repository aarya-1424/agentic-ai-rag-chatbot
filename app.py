import json
import html
import streamlit as st
from datetime import datetime
from rag_graph import ask_question

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Agentic AI — RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown(
    """
    <style>
    .chat-box {
        background: #f6f8fa;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 8px;
        color: #111827;
    }

    .user {
        background: linear-gradient(90deg,#e6f7ff,#ffffff);
        padding:12px;
        border-radius:10px;
        color: #0f172a;
    }

    .assistant {
        background: linear-gradient(90deg,#fff8e6,#ffffff);
        padding:12px;
        border-radius:10px;
        color: #1f2937;
    }

    .small-muted {
        color: #6b7280;
        font-size: 12px;
    }

    .right-col {
        background: #ffffff;
        padding:12px;
        border-radius:8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("🧠 Agentic AI Controls")
    st.markdown("Answers are **strictly grounded** in the Agentic AI eBook.")

    show_context = st.checkbox("Show retrieved context by default", value=True)

    # 🔒 Hard limit = 4
    max_context_chunks = st.slider(
        "Max context chunks",
        min_value=1,
        max_value=4,
        value=4
    )

    confidence_scale = st.selectbox(
        "Confidence display",
        ["Percentage", "Bar"]
    )

    st.markdown("---")

    if st.button("🗑 Clear conversation"):
        st.session_state.chat_history = []
        st.success("Conversation cleared")

    if st.session_state.chat_history:
        conv_json = json.dumps(
            st.session_state.chat_history,
            ensure_ascii=False,
            indent=2
        )
        st.download_button(
            "⬇ Export conversation (JSON)",
            conv_json,
            file_name=f"agenticai_chat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    st.markdown("---")
    st.caption("v1.1 • Agentic AI RAG Chatbot")

# -------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------
left_col, right_col = st.columns([3, 1])

# -------------------------------------------------
# LEFT COLUMN — CHAT
# -------------------------------------------------
with left_col:
    st.title("🤖 Agentic AI — RAG Chatbot")
    st.caption("Powered by LangGraph, FAISS & Groq")

    with st.form("ask_form"):
        question = st.text_area(
            "Ask a question about Agentic AI",
            height=100,
            placeholder="E.g. What is Agentic AI?"
        )
        submit = st.form_submit_button("Ask")

    if submit and question:
        with st.spinner("Thinking..."):
            try:
                result = ask_question(question)
            except Exception as e:
                st.error(f"Error: {e}")
                result = {
                    "answer": "An error occurred.",
                    "confidence": 0.0,
                    "context": []
                }

        conf = result.get("confidence", 0.0)
        conf_val = float(conf) if isinstance(conf, (int, float)) else 0.0

        st.session_state.chat_history.append({
            "question": question,
            "answer": result.get("answer", ""),
            "confidence": conf_val,
            "context": result.get("context", []),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

    st.markdown("### 💬 Conversation")

    if not st.session_state.chat_history:
        st.info("No messages yet. Ask your first question!")
    else:
        for i, chat in enumerate(st.session_state.chat_history):
            # User
            st.markdown(
                f"<div class='chat-box user'><b>🧑 You</b><br>{html.escape(chat['question'])}</div>",
                unsafe_allow_html=True
            )

            # Assistant
            st.markdown(
                f"<div class='chat-box assistant'><b>🤖 Agentic AI</b><br>{html.escape(chat['answer'])}</div>",
                unsafe_allow_html=True
            )

            meta_cols = st.columns([1, 2])
            with meta_cols[0]:
                st.markdown(
                    f"<div class='small-muted'>Time: {chat['timestamp'][:19].replace('T',' ')}</div>",
                    unsafe_allow_html=True
                )
            with meta_cols[1]:
                if confidence_scale == "Bar":
                    st.progress(min(1.0, chat["confidence"]))
                else:
                    st.markdown(
                        f"<div class='small-muted'>Confidence: {chat['confidence']:.2f}</div>",
                        unsafe_allow_html=True
                    )

            st.download_button(
                "⬇ Download answer",
                chat["answer"],
                file_name=f"answer_{i+1}.txt",
                mime="text/plain",
                key=f"dl_{i}"
            )

            if chat["context"]:
                with st.expander("📚 Retrieved Context", expanded=show_context):
                    for j, chunk in enumerate(
                        chat["context"][:max_context_chunks],
                        1
                    ):
                        st.markdown(f"**Chunk {j}:**")
                        st.write(chunk)

            st.markdown("---")

# -------------------------------------------------
# RIGHT COLUMN — SUMMARY
# -------------------------------------------------
with right_col:
    st.markdown("<div class='right-col'>", unsafe_allow_html=True)
    st.subheader("🕘 Recent Q&A")

    if st.session_state.chat_history:
        for chat in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"**Q:** {chat['question'][:60]}...")
            st.markdown(f"**A:** {chat['answer'][:80]}...")
            st.caption(chat["timestamp"][:19].replace("T", " "))
            st.markdown("")
    else:
        st.caption("No recent activity")

    st.markdown("---")
    st.subheader("📖 About sources")
    st.markdown(
        "All answers are generated using retrieved chunks from the "
        "**Agentic AI eBook**. Expand *Retrieved Context* to inspect sources."
    )
    st.markdown("</div>", unsafe_allow_html=True)
