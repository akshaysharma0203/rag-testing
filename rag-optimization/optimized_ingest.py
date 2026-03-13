from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field 
from tqdm import tqdm
from chromadb import PersistentClient
from litellm import completion
from sklearn.manifold import TSNE
from typing import List
import numpy as np
import glob
import plotly.graph_objects as go

load_dotenv(override=True)

model_name="gpt-4.1-nano"
DB_NAME="optimized_db"
embedding_model="text-embedding-3-large"
average_chunk_size = 500
documentcontentlist: list = []


openai=OpenAI()

class DocumentChunk(BaseModel):
    page_content: str
    metadata: dict

class SingleChunk(BaseModel):
    headline: str =Field(description="A brief heading for this chunk, typically  a few words that is mostlikey represent the chunk")
    summary: str =Field(description="Typically a summary  in few sentences to  answer the questions")
    original_text: str = Field(description="The original text of the chunk from the document as it is there in the documents. No changes need to be done in this")

    def return_chunk_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return DocumentChunk(page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,metadata=metadata)

class Chunks(BaseModel):
    chunks: list[SingleChunk]

def fetch_documents() -> List[dict]:
    """fetch the list of documents like DirectoryLoader does"""
    basepath = Path(__file__).resolve().parent.parent
    documentationpath=str(basepath)+"/company_documentation"
    docpath = glob.glob(f"{documentationpath}/**/*.md",recursive=True)
    #print(documentationpath)
    for singledocument in docpath:
        doc_type=str(Path(singledocument).resolve().parent).split('/')[-1:][0]
        #print(doc_type)
        with open(singledocument, 'r', encoding='utf-8') as r:
            documentcontentlist.append({"type":doc_type, "source":singledocument, "text": r.read()})
    return documentcontentlist

# lets do the semantic chunking now after we have fetched the documents from the local

def create_semantic_search_prompt(document: dict):
    numberofchunks = (len(document["text"]))//average_chunk_size + 1
    return f"""
    You take a document and you split the document into overlapping chunks for a KnowledgeBase.

    The document is from the shared drive of a company called Insurellm.
    The document is of type: {document["type"]}
    The document has been retrieved from: {document["source"]}

    A chatbot will use these chunks to answer questions about the company.
    You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
    This document should probably be split into {numberofchunks} chunks, but you can have more or less as appropriate.
    There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

    For each chunk, you should provide a headline, a summary, and the original text of the chunk.
    Together your chunks should represent the entire document with overlap.

    Here is the document:

    {document["text"]}

    Respond with the chunks.
    """
def create_chunks_using_llm(documents):
    chunk = []
    for document in tqdm(documents):
        chunk_message = {"role": "user", "content": create_semantic_search_prompt(document)}
        response = completion(model=model_name, messages=[chunk_message], response_format=Chunks)
        reply = response.choices[0].message.content
        listofpreprocessedchunksforadoc=Chunks.model_validate_json(reply).chunks
        #print(listofpreprocessedchunksforadoc)
        for singlechunk in listofpreprocessedchunksforadoc:
            chunkdata=[singlechunk.return_chunk_result(document)]
            #print(chunkdata)
            chunk.extend(chunkdata)
        #chunk.appent(SingleChunk.return_chunk_result(reply))
        #print(reply)
    print(len(chunk))
    print(chunk)
if __name__ == "__main__":
    create_chunks_using_llm(fetch_documents())