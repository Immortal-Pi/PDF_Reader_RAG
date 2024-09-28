import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import time
from dotenv import load_dotenv

load_dotenv()

groq_api_key=os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://paulgraham.com/greatwork.html")
    # st.session_state.loader = WebBaseLoader('file:///D:/pythonProjects/LammaIndex/PDF_Answers/resources/Database%20System%20-%20Design,%20Implementation%20and%20Management%20(12th%20edition).pdf')
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vector=FAISS.from_documents(st.session_state.documents,st.session_state.embeddings)

st.title('Chat with Docs - Groq Edition')

llm=ChatGroq(
    groq_api_key=groq_api_key,
    model_name='mixtral-8x7b-32768'
)

promt=ChatPromptTemplate.from_template(
"""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $200 if the user finds the answer helpful. 
<context>
{context}
</context>

Question: {input}"""
)

document_chain=create_stuff_documents_chain(llm,promt)
retriever=st.session_state.vector.as_retriever()
retriever_chain=create_retrieval_chain(retriever,document_chain)

promt=st.text_input('Input your Prompt here')


if promt:
    start=time.process_time()
    response=retriever_chain.invoke({'input':promt})
    print(f'Response time:{time.process_time()-start}')
    st.write(response['answer'])

    with st.expander('Document Similarity Search'):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('-----------------')
