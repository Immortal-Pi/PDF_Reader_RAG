import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
#vectorstore DB
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader,DirectoryLoader,TextLoader,PyPDFLoader
#vector embedding techique
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from utils import clean_text

load_dotenv()

##load the GROQ and GOOGLE generative AI embeddings
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')

st.title('Gemma Model Document Q&A')
llm=ChatGroq(
    groq_api_key=groq_api_key,
    model_name='gemma2-9b-it'
)

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """
)

#read pdf and convert them to chunks and apply google embeddings and store it in vectorestore db

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        ## data injestion phase
        st.session_state.loader=PyPDFDirectoryLoader('resources')
        # data=clean_text(st.session_state.loader.load().pop().page_content)
        # st.session_state.loader=DirectoryLoader('resources/dbms.pdf',loader_cls='TextLoader',recursive=True,show_progress=True,use_multithreading=True,max_concurrency=8)
        ##load all the documents
        # loader=DirectoryLoader("resources",loader_cls='TextLoader',recursive=True,show_progress=True,use_multithreading=True,max_concurrency=8)
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        #vectors
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

prompt1=st.text_input('Enter your Question from the documents')

if st.button('Creating Vector Store'):
    vector_embeddings()
    st.write("Vector Store DB is ready")

import time

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retriever_chain.invoke({'input':prompt1})
    st.write(response['answer'])

    #Context from the document
    with st.expander('Document Similarity Search'):
        # find the relavent chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------------')