from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import List
import gradio as gr

load_dotenv(override=True)
model="gpt-4.1-nano"
db_name="vector_db"
RETRIVEL_K=10
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

SYSTEM_PROMPT = """ You are a knowledgeable, friendly assistant representing the company NexaCore Technologies.
You are chatting with a user about NexaCore Technologies.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.

context:
{context}
"""

vectorstore=Chroma(persist_directory=db_name, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(temperature=0,model_name=model)

def fetch_context(question: str) -> List[Document]:
    """
        Retrieve relevant context documents for a question.
    """
    #print(retriever.invoke(question, k=RETRIVEL_K))
    return retriever.invoke(question, k=RETRIVEL_K)
def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question
    
def answer_question(question:str, history):
    retrieved_context = fetch_context(question)
    context = "\n\n".join(rcontext.page_content for rcontext in retrieved_context)
    systemprompt = SYSTEM_PROMPT.format(context=context)
    response = llm.invoke([SystemMessage(content=systemprompt),HumanMessage(content=question)])
    return response.content

if __name__ == '__main__':
    #answer_question("who is tanaka?","")
    gr.ChatInterface(answer_question).launch()
