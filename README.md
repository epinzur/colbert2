# Colbert

This is a playground to use [ColBERT](https://github.com/stanford-futuredata/ColBERT)
It does not use [RAGatouille](https://github.com/bclavie/RAGatouille)

The problem with RAGatouille are
* It does not expose all the ColBERT configurations.
* It is difficult directly integrate with LangChain's Embeddings class

## Gotcha
The latest version of `colbert-ai==0.2.19` or its dependencies require `pyarraow==14.0.0`

Install `faiss-gpu` on CUDA


## what's in the repo
Core of embedding, database loader, retriever and ranker is at this [folder](webserver/webserver/embedding) that includes
* Embedding
* Astra loader
* Astra vector based retriever
An [example](webserver/webserver/example.py) of RAG using ColBERT embedding, Astra DB vector store, retriever (including a default ranker).

### extra
* A web server for embedding service
* Dockerfile of the web embedding service
* Indexing and encoding examples[example] to test on GPU.

## Examples

### LangChain/RagSTACK loader and splitter

Load, split and prepare the documents

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
import os

# pip install pypdf
loader =DirectoryLoader(
    path="./files",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    recursive=True,
)

docs = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500, # colbert doc_maxlen is 220
    chunk_overlap=100,
    length_function=len,
)

splits = text_splitter.split_documents(docs)
title = docs[0].metadata['source']
collections = []

for part in splits:
    collections.append(part.page_content)
```

### Create ColBERT Embeddings

```python
from embedding import ColbertTokenEmbeddings

colbert = ColbertTokenEmbeddings(
    doc_maxlen=220,
    nbits=1,
    kmeans_niters=4,
    nranks=1,
)

passageEmbeddings = colbert.embed_documents(texts=collections, title=title)
```

### Load ColBERT embeddings into Astra DB
Create tables and load embeddings

```python
from embedding import AstraDB
import os

# astra db
astra = AstraDB(
    secure_connect_bundle="./secure-connect-mingv1.zip",
    astra_token=os.getenv("ASTRA_TOKEN"),
    keyspace="colbert128"
)
```

### Retrieval from AstraDB

```python
from embedding import ColbertAstraRetriever

retriever = ColbertAstraRetriever(astraDB=astra, colbertEmbeddings=colbert)
answers = retriever.retrieve("what's the toll free number to call for help?")
```


# Web embedding service

A web embedding [service](webserver/webserver) is implemented to provide ColBERT text embedding over HTTP.

Commands to set up dev environment.
```
cd webserver
poetry install
poetry shell
cd webserver
uvicorn main:app --reload
```

# Next Step
## Parallel loading of embeddings to Astra

## Performance and configuration
* Query performance of transformed one dimensional embedding
* Configuration parameters of ColBERTConfig (chunk size, nbits, kmeans_nitters, nranks on CUDA, bsize, rerank?)
* Two dimensioned index query is implemented in the index.py. Yet I need to measure the relevancy  rank.
