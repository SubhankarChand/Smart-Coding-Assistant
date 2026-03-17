import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic import hub
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# --- 1. Environment Setup ---
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        st.error("Google API Key not found! Please check your .env file.")
        st.stop()

# --- 2. Build the RAG Knowledge Base ---
@st.cache_resource 
def setup_rag_tool():
    loader = DirectoryLoader('./rag_data', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    retriever = vectorstore.as_retriever()
    rag_tool = create_retriever_tool(
        retriever,
        "search_dsa_and_problems",
        "Searches and returns information from custom Data Structures, Algorithms guides, and interview problems. Always use this first for coding questions!"
    )
    return rag_tool

# --- 3. Agent Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, convert_system_message_to_human=True)
search_tool = DuckDuckGoSearchRun()
custom_rag_tool = setup_rag_tool()
tools = [custom_rag_tool, search_tool] 

# Get prompt and inject a custom System Persona
prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.messages[0] = SystemMessage(content="You are a supportive, expert technical coding interviewer. Use your tools to find accurate information. Give hints to help the student learn rather than just giving away the final code immediately.")

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 4. Streamlit Chat UI & Memory ---
st.title("🤖 My Smart Coding Assistant")
st.caption("A LeetCode-style AI agent powered by RAG and Gemini 2.5 Flash")

# Initialize chat memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I'm your coding assistant. Ask me a DSA question, and I'll check my custom guides or search the web!")
    ]

# Display previous messages
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)

# Get user input from the sleek chat bar at the bottom
if user_query := st.chat_input("Ask a coding question..."):
    # Show user message instantly
    st.session_state.messages.append(HumanMessage(content=user_query))
    st.chat_message("user").markdown(user_query)
    
    # Process AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and thinking..."):
            try:
                # Pass the query AND the history to the agent
                chat_history = st.session_state.messages[:-1] 
                response = agent_executor.invoke({
                    "input": user_query,
                    "chat_history": chat_history
                })
                
                # Format output
                raw_output = response['output']
                clean_text = ""
                if isinstance(raw_output, list):
                    for chunk in raw_output:
                        if isinstance(chunk, dict) and "text" in chunk:
                            clean_text += chunk["text"]
                        elif isinstance(chunk, str):
                            clean_text += chunk
                else:
                    clean_text = raw_output
                
                # Display and save AI response
                st.markdown(clean_text)
                st.session_state.messages.append(AIMessage(content=clean_text))
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")