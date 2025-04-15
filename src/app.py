# pip install streamlit langchain langchain-google-genai beautifulsoup4 python-dotenv chromadb 
 
import streamlit as st 
from langchain_core.messages import AIMessage, HumanMessage 
from langchain_community.document_loaders import WebBaseLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain 
 
# Load API key from .env file
load_dotenv() 
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")
 
def get_vectorstore_from_url(url): 
    # get the text in document form 
    loader = WebBaseLoader(url) 
    document = loader.load() 
     
    # split the document into chunks 
    text_splitter = RecursiveCharacterTextSplitter() 
    document_chunks = text_splitter.split_documents(document) 
     
    # create a vectorstore from the chunks 
    vector_store = Chroma.from_documents(document_chunks, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
 
    return vector_store 
 
def get_context_retriever_chain(vector_store): 
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
     
    retriever = vector_store.as_retriever() 
     
    prompt = ChatPromptTemplate.from_messages([ 
      MessagesPlaceholder(variable_name="chat_history"), 
      ("user", "{input}"), 
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation") 
    ]) 
     
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt) 
     
    return retriever_chain 
     
def get_conversational_rag_chain(retriever_chain):  
     
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
     
    prompt = ChatPromptTemplate.from_messages([ 
      ("system", "Answer the user's questions based on the below context:\n\n{context}"), 
      MessagesPlaceholder(variable_name="chat_history"), 
      ("user", "{input}"), 
    ]) 
     
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt) 
     
    return create_retrieval_chain(retriever_chain, stuff_documents_chain) 
 
def get_response(user_input): 
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store) 
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain) 
     
    response = conversation_rag_chain.invoke({ 
        "chat_history": st.session_state.chat_history, 
        "input": user_input 
    }) 
     
    return response['answer'] 
 
# app config 
st.set_page_config(page_title="Chat with websites", page_icon="🤖") 
st.title("Chat with websites") 
 
# Initialize session state for chat history
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = [ 
        AIMessage(content="Hello, I am a bot. How can I help you?"), 
    ] 

# sidebar with URL input only
with st.sidebar: 
    st.header("Website URL")
    website_url = st.text_input("Enter URL to chat with")

# Main chat area
if website_url is None or website_url == "": 
    st.info("Please enter a website URL in the sidebar") 
else: 
    # Initialize vector store if not already done
    if "vector_store" not in st.session_state or st.session_state.get("last_url") != website_url: 
        with st.spinner("Loading website content..."):
            st.session_state.vector_store = get_vectorstore_from_url(website_url)
            st.session_state.last_url = website_url
            st.success(f"Website loaded successfully!")
 
    # user input in main area
    user_query = st.chat_input("Type your message here...") 
    if user_query is not None and user_query != "": 
        with st.spinner("Generating response..."):
            response = get_response(user_query) 
            st.session_state.chat_history.append(HumanMessage(content=user_query)) 
            st.session_state.chat_history.append(AIMessage(content=response)) 
         
    # Display main conversation area
    for message in st.session_state.chat_history: 
        if isinstance(message, AIMessage): 
            with st.chat_message("AI"): 
                st.write(message.content) 
        elif isinstance(message, HumanMessage): 
            with st.chat_message("Human"): 
                st.write(message.content)