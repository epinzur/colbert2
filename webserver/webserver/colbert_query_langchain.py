from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from trulens_eval import Tru, Feedback, TruChain
from trulens_eval.app import App
from trulens_eval.feedback.provider import AzureOpenAI
from trulens_eval.feedback import Groundedness, GroundTruthAgreement
import numpy as np

from typing import List
from dotenv import load_dotenv

load_dotenv()

from embedding import AstraDB, ColbertAstraRetriever, ColbertTokenEmbeddings

import os
import time
import uuid
import json

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

colbert = ColbertTokenEmbeddings(
    doc_maxlen=220,
    nbits=1,
    kmeans_niters=4,
    nranks=1,
)

retriever = ColbertAstraRetriever(astraDB=astra, colbertEmbeddings=colbert)

class ColbertRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = retriever.retrieve(query)
        documents = []
        for result in results:
            body = result['body']
            del result['body']
            documents.append(Document(page_content=body, metadata=result))

        return documents

chatModel = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",
    openai_api_version="2023-05-15",
    model_version="0613",
    temperature=0,
)

colbert_retriever = ColbertRetriever()

prompt_template = """
Answer the question based only on the supplied context. If you don't know the answer, say: "I don't know".
Context: {context}
Question: {question}
Your answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

if os.getenv("TRULENS_DB_CONN_STRING"):
    tru = Tru(database_url=os.getenv("TRULENS_DB_CONN_STRING"))
else:
    tru = Tru()

def get_feedback_functions(pipeline, golden_set):
    # Initialize provider class
    azureOpenAI = AzureOpenAI(deployment_name="gpt-35-turbo-16k")

    context = App.select_context(pipeline)

    # Define a groundedness feedback function
    grounded = Groundedness(groundedness_provider=azureOpenAI)
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons,
                 name="groundedness")
        .on(context.collect()).on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    # Question/answer relevance between overall question and answer.
    f_answer_relevance = (
        Feedback(azureOpenAI.relevance_with_cot_reasons,
                 name="answer_relevance")
        .on_input_output()
    )

    # Question/statement relevance between question and each context chunk.
    f_context_relevance = (
        Feedback(azureOpenAI.qs_relevance_with_cot_reasons,
                 name="context_relevance")
        .on_input().on(context)
        .aggregate(np.mean)
    )

    # GroundTruth for comparing the Answer to the Ground-Truth Answer
    ground_truth_collection = GroundTruthAgreement(
        golden_set, provider=azureOpenAI)
    f_answer_correctness = (
        Feedback(ground_truth_collection.agreement_measure,
                 name="answer_correctness")
        .on_input_output()
    )
    return [f_answer_relevance, f_context_relevance, f_groundedness, f_answer_correctness]

def get_recorder(pipeline, app_id: str, golden_set : [], feedback_mode : str = "deferred"):
    feedbacks = get_feedback_functions(pipeline, golden_set)
    return TruChain(
        pipeline,
        app_id=app_id,
        feedbacks=feedbacks,
        feedback_mode=feedback_mode,
    )

def get_test_data():
    base_path = "../../../../data"

    datasets = {}
    golden_set = []

    for name in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, name)):
            datasets[name] = []
            with open(os.path.join(base_path, name, "rag_dataset.json")) as f:
                examples = json.load(f)['examples']
                for e in examples:
                    datasets[name].append(e["query"])
                    golden_set.append({
                        "query": e["query"],
                        "response": e["reference_answer"],
                    })
                print("Loaded dataset: ", name)
    return datasets, golden_set

# use a short uuid to ensure that multiple experiments with the same name don't collide in the DB
shortUuid = str(uuid.uuid4())[9:13]
datasets, golden_set = get_test_data()

chain = (
    {"context": colbert_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chatModel
    | StrOutputParser()
)

for dataset_name in datasets:
    app_id = f"ColBERT_AstraDB#{shortUuid}#{dataset_name}"
    print(f"Starting processing App: {app_id}")
    tru_recorder = get_recorder(chain, app_id, golden_set)
    for query in datasets[dataset_name]:
        try:
            with tru_recorder as recording:
                chain.invoke(query)
        except:
            print(f"Query: '{query}' caused exception, skipping.")

astra.close()
