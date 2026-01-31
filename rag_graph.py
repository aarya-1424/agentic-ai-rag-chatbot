import os
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


# -----------------------------
# CONFIG
# -----------------------------
FAISS_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"


# -----------------------------
# STATE DEFINITION
# -----------------------------
class RAGState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    confidence: float


# -----------------------------
# LOAD VECTOR STORE
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 4,
        "score_threshold": 0.3
    }
)


# -----------------------------
# LOAD GROQ LLM
# -----------------------------
llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# -----------------------------
# RETRIEVE NODE
# -----------------------------
def retrieve(state: RAGState):
    docs = retriever.invoke(state["question"])
    return {"documents": docs}


# -----------------------------
# GENERATE NODE
# -----------------------------
def generate(state: RAGState):
    docs = state["documents"]

    # 🚫 No relevant documents
    if not docs:
        return {
            "answer": "Answer not available in the provided document.",
            "confidence": 0.0
        }

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate.from_template(
        """
You are an AI assistant.
Answer the question ONLY using the context below.
If the answer is not present, say:
"Answer not available in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    response = llm.invoke(
        prompt.format(
            context=context,
            question=state["question"]
        )
    )

    # 🎯 Confidence based on retrieval strength
    confidence = round(min(1.0, 0.3 + 0.15 * len(docs)), 2)

    return {
        "answer": response.content.strip(),
        "confidence": confidence
    }


# -----------------------------
# LANGGRAPH WORKFLOW
# -----------------------------
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_app = workflow.compile()


# -----------------------------
# PUBLIC HELPER (USED BY app.py)
# -----------------------------
def ask_question(question: str):
    result = rag_app.invoke({"question": question})

    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "context": [doc.page_content for doc in result.get("documents", [])]
    }


# -----------------------------
# CLI TEST (OPTIONAL)
# -----------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break

        output = ask_question(q)
        print("\nANSWER:\n", output["answer"])
        print("\nCONFIDENCE:", output["confidence"])
