from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from embedding import AstraDB
import os
import time

start_time = time.time()
os.environ["ASTRA_DB_ENDPOINT"] = os.environ.get("ASTRA_DB_ENDPOINT_COLBERT")
os.environ["ASTRA_DB_TOKEN"] = os.environ.get("ASTRA_DB_TOKEN_COLBERT")
os.environ["SECURE_CONNECT_BUNDLE"] = os.environ.get("SECURE_CONNECT_BUNDLE_COLBERT")

# astra db
astra = AstraDB(
    secure_connect_bundle=os.getenv("SECURE_CONNECT_BUNDLE"),
    astra_token=os.getenv("ASTRA_DB_TOKEN"),
    keyspace="default_keyspace",
)

astra.ping()

print("astra db is connected")

# pip install pypdf
loader = DirectoryLoader(
    path = "../../../../data",
    glob="*/source_files/*.pdf",
    loader_cls=PyPDFLoader,
    recursive=True,
)

docs = loader.load()

print(f"Loaded {len(docs)} documents")

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

print(f"title {title}, doc size {len(docs)} splitted size {len(collections)}")

from embedding import ColbertTokenEmbeddings

colbert = ColbertTokenEmbeddings(
    doc_maxlen=220,
    nbits=1,
    kmeans_niters=4,
    nranks=1,
)

passageEmbeddings = colbert.embed_documents(texts=collections, title=title)

print(f"passage embeddings size {len(passageEmbeddings)}")

# astra insert colbert embeddings
astra.insert_colbert_embeddings_chunks(passageEmbeddings)

duration = time.time() - start_time
print(f"It took {duration} seconds to load the documents via colBERT.")
