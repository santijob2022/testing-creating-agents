
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams,Filter
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings

class QdrantVectorStoreManager:
    def __init__(self, host="localhost", port=6333,
                 embedding_model="text-embedding-3-small",
                 vector_size=1536):
        self.client = QdrantClient(host=host, port=port)
        self.embedding = OpenAIEmbeddings(model=embedding_model)
        self.vector_size = vector_size

    def get_vectorstore(self, collection_name: str) -> Qdrant:
        if not self.client.collection_exists(collection_name):
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
        vectorstore = Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=self.embedding,
        )
        vectorstore.qdrant_client = self.client
        vectorstore.embedding = self.embedding
        return vectorstore

    def delete_all_entries(self, collection_name: str) -> None:
        self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(must=[])
        )