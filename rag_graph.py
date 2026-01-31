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

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# -----------------------------
# LOAD GROQ LLM
# -----------------------------
llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# -----------------------------
# RETRIEVAL STEP
# -----------------------------
def retrieve(state: RAGState):
    docs = retriever.invoke(state["question"])
    return {"documents": docs}


# -----------------------------
# GENERATION STEP
# -----------------------------
def generate(state: RAGState):
    context = "\n\n".join([doc.page_content for doc in state["documents"]])

    prompt = PromptTemplate.from_template(
        """
You are an AI assistant.
Answer the question ONLY using the provided context.
If the answer is not present in the context, say:
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

    confidence = min(1.0, len(state["documents"]) / 4)

    return {
        "answer": response.content,
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
# HELPER FUNCTION
# -----------------------------
def ask_question(question: str):
    result = rag_app.invoke({"question": question})

    return {
        "answer": result["answer"],
        "context": [doc.page_content for doc in result["documents"]],
        "confidence": result["confidence"]
    }


# -----------------------------
# TEST (CLI)
# -----------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break

        output = ask_question(q)

        print("\nANSWER:\n", output["answer"])
        print("\nCONFIDENCE:", output["confidence"])
