# 🤖 Agentic AI — RAG Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot built using **LangGraph**, **FAISS**, and **Groq LLM**, designed to answer questions strictly from a provided PDF document.

This project demonstrates how Agentic AI systems retrieve relevant knowledge before generating responses, ensuring grounded and reliable answers.

---

## 🚀 Features

- 📄 PDF ingestion & chunking
- 🔍 Semantic search using FAISS vector database
- 🧠 Embeddings with Sentence-Transformers (MiniLM)
- 🔗 LangGraph-based RAG workflow
- 🤖 Groq LLM integration (LLaMA 3.1)
- 📊 Confidence scoring based on retrieval relevance
- 🧾 Fallback when answer is not in the document
- 🎨 Interactive Streamlit UI
- 📚 Retrieved context inspection

---

## 🏗️ Tech Stack

- **Python**
- **LangChain**
- **LangGraph**
- **FAISS**
- **Groq API**
- **HuggingFace Embeddings**
- **Streamlit**

---

## 📁 Project Structure
# 🤖 Agentic AI — RAG Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot built using **LangGraph**, **FAISS**, and **Groq LLM**, designed to answer questions strictly from a provided PDF document.

This project demonstrates how Agentic AI systems retrieve relevant knowledge before generating responses, ensuring grounded and reliable answers.

---

## 🚀 Features

- 📄 PDF ingestion & chunking
- 🔍 Semantic search using FAISS vector database
- 🧠 Embeddings with Sentence-Transformers (MiniLM)
- 🔗 LangGraph-based RAG workflow
- 🤖 Groq LLM integration (LLaMA 3.1)
- 📊 Confidence scoring based on retrieval relevance
- 🧾 Fallback when answer is not in the document
- 🎨 Interactive Streamlit UI
- 📚 Retrieved context inspection

---

## 🏗️ Tech Stack

- **Python**
- **LangChain**
- **LangGraph**
- **FAISS**
- **Groq API**
- **HuggingFace Embeddings**
- **Streamlit**

---

## 📁 Project Structure
agentic-ai-rag-chatbot/
├── app.py # Streamlit frontend
├── rag_graph.py # RAG logic using LangGraph
├── ingest.py # PDF ingestion & FAISS indexing
├── data/ # PDF files
├── requirements.txt
├── README.md
├── .gitignore
└── .env # Environment variables (not committed)


---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/agentic-ai-rag-chatbot.git
cd agentic-ai-rag-chatbot

2️⃣ Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add environment variables
Create a .env file:
GROQ_API_KEY=your_groq_api_key_here

📥 Ingest PDF
Place your PDF inside the data/ folder, then run:
python ingest.py

▶️ Run the App
streamlit run app.py