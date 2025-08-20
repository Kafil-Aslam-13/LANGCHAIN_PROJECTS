## RAG Q&A Conversation with PDF Including Chat History
from langchain.chains.retrieval import create_retrieval_chain 
from langchain.chains.history_aware_retriever import create_history_aware_retriever #will be used to create retriever with chat history functionalities
from langchain.chains.combine_documents import create_stuff_documents_chain # create entire document and send it to context
from langchain_chroma import Chroma 
import streamlit as st

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_LANGCHAIN_TOKEN"]=os.getenv('HF_LANGCHAIN_TOKEN')
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## set up our streamlit app 
st.title("Conversational Rag with PDF uploads and Chat History")
st.write("Upload PDF and chat with the content")

## Input the groq api key
groq_api_key=st.text_input("Enter the groq api key",type="password")


## check if groq api is provided
if groq_api_key:
    llm=ChatGroq(model_name="llama-3.1-8b-instant",groq_api_key=groq_api_key)

    ## chat interface
    session_id=st.text_input("Session ID",value="default_session")
    ## statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}
    
    uploaded_files=st.file_uploader("choose a pdf file",type="pdf",accept_multiple_files=True)
    ## process  my uploaded files
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf = f"./temp_{uploaded_file.name}"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

        ## split and create embeddings
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vectorstore.as_retriever()

        contextualized_q_system_prompt=(
            "Given chat history and lates user question"
            "which night reference context in the chat history"
            "formulate a standalone question which can be understood"
            "without the chat history. Do NOT answer the question"
            "just reformulate it if needed and otherwise return as it is"
        )

        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualized_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        ## history retriever with memory
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        ## Answwer Question prompt

        system_prompt=(
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer"
            "the question. If you don't know the answer ,say that you"
            "don't know. Use Three sentences maximum and keep the "
            "answer concise"
            "\n\n"
            "{context}"
        )

        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        ## creating question answer chain

        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        

        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input=st.text_input("your question")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={"configurable":{"session_id":session_id}},
            )

            st.write(st.session_state.store)
            st.write("Assistant",response['answer'])
            st.write("CHAT History",session_history.messages)

else:
    st.warning("Please enter your session key")
    


