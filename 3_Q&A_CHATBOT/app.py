import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

## langsmith tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With OpenAI"

## defining chat prompt template

prompt=ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant.Please response to user queries"),
    ("user","Question:{question}")
])

## inside this below fn we interact with model and should give response

def generate_response(question,api_key,llm ,temperature,max_tokens):
    openai.api_key=api_key
    llm=ChatOpenAI(model=llm,temperature=temperature,max_tokens=max_tokens)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer


## title of the app
st.title("Enhanced QandA chatbot with OpenAI")

## Sidebar for  settings
st.sidebar.title('Settings')
api_key=st.sidebar.text_input("Enter your Open Ai Api Key",type="password")
## drop down to select various llm models
llm=st.sidebar.selectbox("select an openai model",["gpt-4o","gpt-4-turbo","gpt-4"])

## adjusting response parameter like temp and max tokens

temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

## main interface for user ip
st.write("ASK ANY QUESTION")
user_input=st.text_input("you:")

if user_input and api_key:
    response=generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
elif user_input:
    st.warning("Please enter Open AIAPI KEY IN SIDEBAR")
else:
    st.write("please provide question")
