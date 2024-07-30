import streamlit as st 
import os
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()
os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY") 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = genai.GenerativeModel('gemini-pro') 
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# prompt template 
prompt = ChatPromptTemplate.from_template(

    '''You need to provide the answers based on the context provided.
    Try to give the most accurate respone based on the question and think before you answer
    <context>
    {context}
    <context>
    question: {input}
    '''
)

# using streamlit.session_state to store the variables 
# creating vector embeddings 

def vector_embeddings():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings=embeddings
        # load the pdf 
        st.session_state.loader=PyPDFDirectoryLoader("C:\Users\mahik\Documents\GitHub\Langchain_RAG_app\600+ Data Science Interview Questions.pdf")
        st.session_state.docs=st.session_state.loader.load()
        # text splitter 
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        # PDF is 160 pages , let's only consider the first 70 pages
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:70])
        # vector store 
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

# title in streamlit 
st.title("Langchain_RAG_doument_app with gemini-pro")
user_input = st.text_input("Please enter your question from the document:")