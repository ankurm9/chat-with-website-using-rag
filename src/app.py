# pip install streamlit langchain langchain-google-genai beautifulsoup4 python-dotenv chromadb

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

# Load API key (from secrets for Streamlit Cloud or local env as fallback)
google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("❌ Please set GOOGLE_API_KEY in Streamlit Secrets or a .env file.")
    st.stop()

# ========== Functions ==========
def get_vectorstore_from_url(url):
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter()
        chunks = text_splitter.split_documents(documents)

        vector_store = FAISS.from_documents(
            chunks, GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        )
        return vector_store
    except Exception as e:
        st.error(f"Error loading and processing website: {e}")
        st.stop()


def get_context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    return create_history_aware_retriever(llm, retriever, prompt)


def get_conversational_rag_chain(retriever_chain):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, doc_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response["answer"]

# ========== Streamlit App Config ==========
st.set_page_config(page_title="Chat with Websites", page_icon="🤖")
st.title("🤖 Chat with Websites")

# ========== Sidebar ==========
with st.sidebar:
    st.header("Website URL")
    website_url = st.text_input("Enter URL to chat with")

# ========== Session State ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?")
    ]

# ========== Main Chat ==========
if not website_url:
    st.info("ℹ️ Please enter a website URL in the sidebar.")
else:
    if "vector_store" not in st.session_state or st.session_state.get("last_url") != website_url:
        with st.spinner("🔄 Loading website content..."):
            st.session_state.vector_store = get_vectorstore_from_url(website_url)
            st.session_state.last_url = website_url
            st.success("✅ Website loaded successfully!")

    user_query = st.chat_input("Type your message here...")
    if user_query:
        with st.spinner("🤖 Generating response..."):
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
