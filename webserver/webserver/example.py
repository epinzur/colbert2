from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

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

print(f"title {title}, doc size {len(docs)} splitted size {len(collections)}")

from embedding import ColbertTokenEmbeddings

colbert = ColbertTokenEmbeddings(
    doc_maxlen=220,
    nbits=1,
    kmeans_niters=4,
)

passageEmbeddings = colbert.embed_documents(texts=collections, title=title)

print(f"passage embeddings size {len(passageEmbeddings)}")

for pEmbd in passageEmbeddings:

    print(f"passage embedding title {pEmbd.title()} size {len(pEmbd.get_all_token_embeddings())}")
    print(f"passagen embedding id {pEmbd.id()}")
    for tokenEmbd in pEmbd.get_all_token_embeddings():
        print(f"    token embedding id {tokenEmbd.id} parent {tokenEmbd.parent_id} size {len(tokenEmbd.get_embeddings())}")


from embedding import AstraDB
import os

# astra db
astra = AstraDB(
    secure_connect_bundle="./secure-connect-mingv1.zip",
    astra_token=os.getenv("ASTRA_TOKEN"),
    keyspace="colbert2",
    verbose=True,
)

astra.ping()

print("astra db is connected")

# astra insert colbert embeddings
astra.insert_colbert_embeddings_chunks(
    embeddings=passageEmbeddings, delete_existed_passage=True)

from embedding import ColbertAstraRetriever

retriever = ColbertAstraRetriever(astraDB=astra, colbertEmbeddings=colbert)
answers = retriever.retrieve("what's the toll free number to call for help?")
for a in answers:
    print(f"answer rank {a['rank']} score {a['score']}, answer is {a['body']}\n")

# LangChain retriever
print(retriever.get_relevant_documents("what's the toll free number to call for help?"))

astra.close()

