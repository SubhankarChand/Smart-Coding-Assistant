import os

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS

from langchain_classic.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Smart Coding Assistant",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    google_api_key = st.secrets.get("GOOGLE_API_KEY")

if not google_api_key:
    st.error(
        "Google API Key not found. "
        "For local development, add it to a .env file. "
        "For Streamlit Cloud, add it under App Settings → Secrets."
    )
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key


# -----------------------------
# Build RAG
# -----------------------------
@st.cache_resource(show_spinner=False)
def setup_rag_tool():

    loader = DirectoryLoader(
        "rag_data",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    splits = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever()

    rag_tool = create_retriever_tool(
        retriever,
        "search_dsa_and_problems",
        "Search DSA notes and interview questions.",
    )

    return rag_tool


# -----------------------------
# Create Agent
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

search_tool = DuckDuckGoSearchRun()

rag_tool = setup_rag_tool()

tools = [
    rag_tool,
    search_tool,
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert coding interview assistant.

Always search the RAG documents first for DSA questions.

If the answer is not available, use web search.

Provide hints before giving the final solution.
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
)

# -----------------------------
# UI
# -----------------------------
st.title("🤖 Smart Coding Assistant")

st.caption(
    "Powered by Gemini + LangChain + RAG"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(
            content="Hello! Ask me any coding or DSA question."
        )
    ]

for message in st.session_state.messages:

    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

    else:
        with st.chat_message("user"):
            st.markdown(message.content)


user_query = st.chat_input("Ask a coding question...")

if user_query:

    st.session_state.messages.append(
        HumanMessage(content=user_query)
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = agent_executor.invoke(
                    {
                        "input": user_query,
                        "chat_history": st.session_state.messages[:-1],
                    }
                )

                answer = response["output"]

                if isinstance(answer, list):
                    final_answer = ""

                    for item in answer:

                        if isinstance(item, dict):
                            final_answer += item.get(
                                "text",
                                "",
                            )

                        else:
                            final_answer += str(item)

                else:
                    final_answer = str(answer)

                st.markdown(final_answer)

                st.session_state.messages.append(
                    AIMessage(content=final_answer)
                )

            except Exception as e:
                st.error(str(e))