# 🤖 Smart Coding Assistant

An AI-powered coding assistant that helps students prepare for coding interviews using **Retrieval-Augmented Generation (RAG)**, **Google Gemini**, and **LangChain**. The assistant searches a custom DSA knowledge base before performing live web searches, providing accurate and context-aware interview guidance.

## 🌐 Live Demo

**Try the application here:**

https://smart-coding-assistant-cucqzselq3p8jngpgun6a3.streamlit.app/

---

## 📌 Overview

Smart Coding Assistant is designed to simulate an intelligent interview mentor. Instead of relying only on a large language model, it first retrieves relevant information from a custom-built Data Structures and Algorithms knowledge base using semantic search. If the required information is unavailable, it automatically performs a web search to provide the most relevant answer.

The application maintains conversation history, allowing users to ask follow-up questions naturally.

---

## ✨ Features

- 📚 **RAG-Powered Knowledge Base**
  - Searches custom DSA and interview preparation documents using FAISS vector search.

- 🤖 **Google Gemini Integration**
  - Uses Gemini 2.5 Flash for fast and accurate conversational responses.

- 🌍 **Automatic Web Search**
  - Falls back to DuckDuckGo search when information is unavailable in the local knowledge base.

- 💬 **Conversational Memory**
  - Maintains chat history for context-aware responses.

- ⚡ **Fast Semantic Search**
  - Generates embeddings using Gemini Embedding-001 and retrieves the most relevant content.

- 🎯 **Interview-Oriented Responses**
  - Explains concepts, provides hints, and guides users toward solutions instead of only giving final answers.

- 🖥️ **Interactive Streamlit Interface**
  - Clean and responsive chat-based user interface.

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Frontend | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Gemini Embedding-001 |
| AI Framework | LangChain |
| Vector Database | FAISS |
| Web Search | DuckDuckGo |
| Environment | Python 3.11 |

---

## 📂 Project Structure

```text
Smart-Coding-Assistant/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── rag_data/
│   ├── *.txt
│
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SubhankarChand/Smart-Coding-Assistant.git

cd Smart-Coding-Assistant
```

---

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv

venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=your_google_api_key
```

---

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## 🚀 Deployment

This project is deployed on **Streamlit Community Cloud**.

During deployment:

- Add your `GOOGLE_API_KEY` under **App Settings → Secrets**.
- Do **not** upload your `.env` file.
- Use Python 3.11 (`runtime.txt`).

---

## 📷 Application Workflow

```
User Question
      │
      ▼
LangChain Agent
      │
      ├────────► RAG (FAISS + DSA Documents)
      │
      └────────► DuckDuckGo Search
                  │
                  ▼
            Google Gemini
                  │
                  ▼
         Final AI Response
```

---

## 🎯 Future Improvements

- PDF upload and indexing
- Code execution sandbox
- Syntax highlighting
- Multiple programming language support
- Conversation export
- Voice input
- Persistent vector database
- User authentication
- Personalized interview roadmap

---

## 👨‍💻 Author

**Subhankar Chand**

- GitHub: https://github.com/SubhankarChand
- LinkedIn: https://www.linkedin.com/in/subhankar-chand-9708a4258

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

Feedback and contributions are always welcome!
