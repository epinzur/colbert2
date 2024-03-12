from embedding import ColbertTokenEmbeddings

from embedding import AstraDB
from embedding import ColbertAstraRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List

# max similarity between a query vector and a list of embeddings
# The function returns the highest similarity score (i.e., the maximum dot product value) between the query vector and any of the embedding vectors in the list.


class ColbertAstraLangChainRetriever(BaseRetriever):
    retriever: ColbertAstraRetriever = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        astraDB: AstraDB,
        colbertEmbeddings: ColbertTokenEmbeddings,
        **kwargs
    ):
        # First, instantiate the retriever
        # Now, it's safe to call super().__init__ because self.retriever exists
        super().__init__(astra=astraDB, colbertEmbeddings=colbertEmbeddings, **kwargs)
        self.retriever = ColbertAstraRetriever(astraDB=astraDB, colbertEmbeddings=colbertEmbeddings)
        # initialize pydantic base model
 
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        answers = self.retriever.retrieve(query)
        documents = [Document(metadata={'title': d['title'], 'score': d['score'], 'rank': d['rank']}, page_content=d['body']) for d in answers]
        return documents
