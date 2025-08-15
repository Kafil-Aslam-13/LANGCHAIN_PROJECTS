## important libraries which i am going to use in this project
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()
# load groq api
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')

##
# creating our LLM model

llm=ChatGroq(model_name="llama-3.1-8b-instant",groq_api_key=groq_api_key)
# this will be a little different bcz here we need some context information
prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on provided context
    Please provide the most accurate response based on question
    <context>
    {context}
    </context>
    Question:{input}

    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings(model='mxbai-embed-large:latest')
        st.session_state.loader=PyPDFDirectoryLoader("research_papers") ## data ingestion step
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

user_prompt=st.text_input("enter your query from research papaer")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector Database is ready")

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f'response time:{time.process_time()-start}')

    st.write(response['answer'])

    # with streamlit expander
    with st.expander("Document similarity search "):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('---------------------------')
