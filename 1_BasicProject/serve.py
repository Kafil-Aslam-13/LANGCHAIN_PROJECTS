from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes # this add routes will help create your apis
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv('GROQ_API_KEY')

# initialize model
model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

# creating prompt template
system_template="Translate the following into {language}:"

prompt=ChatPromptTemplate.from_messages([
    ("system",system_template),
    ("user","{text}")
    ])

## string op parser
parser=StrOutputParser()

#create chain 
chain=prompt|model|parser


## App definition
app=FastAPI(title="Langchain Server",version="1.0",description="Simple API server using Langchain interfaces") # its like flask but with additional functionality

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)
    