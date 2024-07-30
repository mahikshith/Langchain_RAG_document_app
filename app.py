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
os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY") 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = genai.GenerativeModel('gemini-pro') # try with gemini flash
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
        st.session_state.loader=PyPDFDirectoryLoader(r"C:\Users\mahik\Documents\GitHub\Langchain_RAG_app\trail_documents")
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

# button to create vector embeddings 
if st.button("Create document embeddings"):
    vector_embeddings()
    st.write("Vector store has been created")

if user_input:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    response = retrieval_chain.invoke({"question": user_input})

    # show the answer in streamlit 
    st.write(response['answer'])

# for remaining similar pages -- using similarity search 
# using streamlit expander
with st.expander("Similar results :"):
    for i,doc in enumerate(response['context']):
        st.write(doc.page_content)
        st.write('****************************')

# we can use streamlit document uploader to upload the pdf 
