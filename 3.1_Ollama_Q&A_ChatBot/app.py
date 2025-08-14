from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
import streamlit as st


import os
from dotenv import load_dotenv
load_dotenv()

## langsmith tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With Ollama"


## prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.Please response to user queries"),
        ("user","Question:{question}")
    ]
)


## inside this below fn we interact with model and should give response

def generate_response(question,engine ,temperature,max_tokens):
    llm=OllamaLLM(model=engine,temperature=temperature,max_tokens=max_tokens)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer


## title of the app
st.title("Enhanced QandA chatbot with Ollama")

## Sidebar for  settings
st.sidebar.title('Settings')

## drop down to select various llm models
engine=st.sidebar.selectbox("select an openai model",["gemma2:2b","llama3:8b"])

## adjusting response parameter like temp and max tokens

temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

## main interface for user ip
st.write("ASK ANY QUESTION")
user_input=st.text_input("you:")

if user_input:
    response=generate_response(user_input,engine,temperature,max_tokens)
    st.write(response)
else:
    st.write("please provide question")
