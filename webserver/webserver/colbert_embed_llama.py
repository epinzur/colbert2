from dotenv import load_dotenv

load_dotenv()

from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline

from llama_index.core import SimpleDirectoryReader

from embedding import AstraDB
import os
import time

from embedding import ColbertTokenEmbeddings

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

print("Astra DB is connected.\nStarting to collect documents.")

reader = SimpleDirectoryReader(
    "../../../../data",
    recursive=True,
    required_exts=[".pdf", ".md", ".txt", ".html"]
)
docs = reader.load_data()

print(f"Loaded {len(docs)} documents")

splitter = TokenTextSplitter(chunk_size=160, chunk_overlap=50) # colbert doc_maxlen is 220 tokens
pipeline = IngestionPipeline(transformations=[splitter])

nodes = pipeline.run(documents=docs)

print(f"Split into {len(nodes)} nodes")

file_text_chunks = {}

for node in nodes:
    title = os.path.normpath(node.extra_info["file_name"])
    if title not in file_text_chunks:
        file_text_chunks[title] = []
    file_text_chunks[title].append(node.text)


print(f"found {len(file_text_chunks)} files inside the nodes")

print("starting to make colbert embeddings")

colbert = ColbertTokenEmbeddings(
    doc_maxlen=220,
    nbits=1,
    kmeans_niters=4,
    nranks=1,
)

for title in file_text_chunks:
    texts = file_text_chunks[title]

    print(f"starting embedding {title} that has {len(texts)} chunks")

    passageEmbeddings = colbert.embed_documents(texts=texts, title=title)

    print(f"passage embeddings size {len(passageEmbeddings)}.\nStarting to insert into Astra DB.")

    # astra insert colbert embeddings
    astra.insert_colbert_embeddings_chunks(passageEmbeddings)

    print(f"Insert completed for {title}\n")

duration = time.time() - start_time
print(f"It took {duration} seconds to load the documents via colBERT.")

astra.close()
