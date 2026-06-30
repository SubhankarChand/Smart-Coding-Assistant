# 🤖 Smart Coding Assistant (RAG-Powered AI Agent)

A specialized, LeetCode-style AI coding assistant built to help engineering students prepare for technical interviews. This application utilizes a Retrieval-Augmented Generation (RAG) architecture to search custom Data Structures and Algorithms (DSA) study guides before falling back to live web searches.

## 🚀 Features
* **Custom Knowledge Base (RAG):** Answers are grounded in custom `.txt` files using FAISS vector embeddings, ensuring specific, curated advice for interview problems.
* **Multi-Tool AI Agent:** The agent decides dynamically whether to query local documents or search the live internet (via DuckDuckGo) based on the user's prompt.
* **Conversational Memory:** Maintains context across multiple chat turns, allowing for follow-up questions and code refinements.
* **Modern UI:** Built with Streamlit's chat interface for a clean, intuitive user experience.

## 🛠️ Tech Stack
* **Language:** Python
* **Framework:** Streamlit
* **LLM Orchestration:** LangChain & LangChain Agents
* **Models:** Google Gemini 2.5 Flash (Chat) & Gemini-Embedding-001 (Embeddings)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Tools:** DuckDuckGo Web Search

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   cd your-repo-name
   
2. **Create a virtual environment and install dependencies:**
  ```bash
   python -m venv venv
  ```
3. **Activate venv**
 ```bash
  (Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate)
 ```
## Downloads Requirements library
 ```bash
  pip install -r requirements.txt
 ```
## Set up API Keys:
Create a .env file in the root directory and add your Google Gemini API key:
```bash
  GOOGLE_API_KEY="your_api_key_here"
```

## Run the Application:
```bash
  streamlit run app.py
```
