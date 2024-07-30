import streamlit as st 
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found in the environment variables.")
else:
    os.environ['GOOGLE_API_KEY'] = api_key
    genai.configure(api_key=api_key)

try:
    llm = genai.GenerativeModel('gemini-pro')
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
except Exception as e:
    st.error(f"Error initializing models: {e}")

prompt = ChatPromptTemplate.from_template(
    '''You need to provide the answers based on the context provided.
    Try to give the most accurate response based on the question and think before you answer
    <context>
    {context}
    <context>
    question: {input}
    '''
)

def vector_embeddings():
    try:
        if 'vectors' not in st.session_state:
            st.session_state.embeddings = embeddings
            st.session_state.loader = PyPDFDirectoryLoader(r"C:\Users\mahik\Documents\GitHub\Langchain_RAG_app\trail_documents")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:70])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    except Exception as e:
        st.error(f"Error in vector embeddings: {e}")

st.title("Langchain_RAG_document_app with gemini-pro")
user_input = st.text_input("Please enter your question from the document:")

if st.button("Create document embeddings"):
    vector_embeddings()
    st.write("Vector store has been created")

if user_input:
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"question": user_input})
        st.write(response.get('answer', "No answer found."))
    except Exception as e:
        st.error(f"Error in retrieval chain: {e}")

with st.expander("Similar results :"):
    try:
        for i, doc in enumerate(response.get('context', [])):
            st.write(doc.page_content)
            st.write('****************************')
    except Exception as e:
        st.error(f"Error displaying similar results: {e}")

# Ensure correct path for document uploader
if not os.path.exists(r"C:\Users\mahik\Documents\GitHub\Langchain_RAG_app\trail_documents"):
    st.error("The specified PDF file path does not exist.")
