import os
import glob
import gradio as gr
import tiktoken
import numpy as np 
import plotly.graph_objects as go
from typing import List
from sklearn.manifold import TSNE
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
print(f"key found and the value of the key is{openai_api_key[:4]}") if openai_api_key else print("key is found")
model="gpt-4o-nanno"
db_name="vector_db"


def read_files() -> str:
    all_documents_content = ""
    #all_resume = {}
    filenames = glob.glob("company_documentation/company/**/*.md", recursive=True)
    print(f"total number of files found is {len(filenames)}")
    for filename in filenames:
        #name = Path(filename).stem.split('_')[-1]
        with open(filename, 'r', encoding="utf-8") as f:
            #all_resume[name.lower()] = f.read()
            all_documents_content += f.read()
            all_documents_content += "\n\n"
    print(f"total number of characters in the documers are {len(all_documents_content):,}")

    # lets calculate the total number of token in the document
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(all_documents_content)
    print(f"total tokens on this documents for model {model} are {len(tokens):,}")
    return all_documents_content

def load_and_chunk() -> List:
    # loading the file contents using the loader and adding doc_type in metadata
    folders = glob.glob("company_documentation/company/*")
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding':'utf-8'})
        loaded_folder = loader.load()
        #loaded_folder is obejct of Document from langchain_core.documents
        #print(loaded_folder)
        for doc in loaded_folder:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
            #print (doc)
        print(f"length of loaded {doc_type} documents {len(documents)}")

    # now lets split the content into chunks so that later those chunks cna be used to vectorize

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"No of chunks created {len(chunks)}")
    return chunks
    # print(f"first chunk is \n \n {chunks[0]}")    

def make_vectors_and_load():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    chunks = load_and_chunk()
    print(f"-----No of chunks created {len(chunks)}")

    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
    print(f"vector store is created with {vector_store._collection.count()} documents")
    # below line calculates the dimensions of the vector
    sample_embedding = vector_store._collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {vector_store._collection.count():,} vectors with {dimensions:,} dimensions in the vector store")
    visualizein2d(vector_store)
    visualizein3d(vector_store) 


def visualizein2d(vectorstore: Chroma):
    result = vectorstore._collection.get(include=['embeddings', 'documents', 'metadatas'])
    vectors = np.array(result['embeddings'])
    documents = result['documents']
    metadatas = result['metadatas']
    doc_types = [metadata['doc_type'] for metadata in metadatas]
    colors = [['blue', 'green', 'red', 'orange','black'][['products', 'employees', 'contracts', 'company','careers'].index(t)] for t in doc_types]
    # We humans find it easier to visalize things in 2D!
    # Reduce the dimensionality of the vectors to 2D using t-SNE
    # (t-distributed stochastic neighbor embedding)

    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    # Create the 2D scatter plot
    fig = go.Figure(data=[go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
        hoverinfo='text'
    )])

    fig.update_layout(title='2D Chroma Vector Store Visualization',
        scene=dict(xaxis_title='x',yaxis_title='y'),
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    fig.show()

def visualizein3d(vectorstore: Chroma):
    result = vectorstore._collection.get(include=['embeddings', 'documents', 'metadatas'])
    vectors = np.array(result['embeddings'])
    documents = result['documents']
    metadatas = result['metadatas']
    doc_types = [metadata['doc_type'] for metadata in metadatas]
    colors = [['blue', 'green', 'red', 'orange','black'][['products', 'employees', 'contracts', 'company','careers'].index(t)] for t in doc_types]
    
    tsne = TSNE(n_components=3, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='3D Chroma Vector Store Visualization',
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
        width=900,
        height=700,
        margin=dict(r=10, b=10, l=10, t=40)
    )

    fig.show()

if __name__ == "__main__":
    make_vectors_and_load()