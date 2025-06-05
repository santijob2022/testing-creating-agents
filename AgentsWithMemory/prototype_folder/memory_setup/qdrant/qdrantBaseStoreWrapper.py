# qdrantBaseStoreWrapper.py

from typing import Any, Tuple
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
import uuid
from langchain.schema import Document

from .baseStoreInterface import BaseStore

class QdrantBaseStoreWrapper(BaseStore):
    def __init__(self, vectorstore: Qdrant, namespace: Tuple[str, ...]):
        self.vectorstore = vectorstore
        self.namespace = namespace
        self.client = vectorstore.qdrant_client
        self.embedding = vectorstore.embedding
        self.collection_name = vectorstore.collection_name

    def _resolve_namespace(self, config: dict) -> str:
        resolved = tuple(
            config.get(part.strip("{}"), part) if part.startswith("{") and part.endswith("}") else part
            for part in self.namespace
        )
        return "/".join(resolved)

    def _build_filter(self, namespace: str, key: str = None):
        conditions = [
            FieldCondition(key="namespace", match=MatchValue(value=namespace))
        ]
        if key:
            conditions.append(FieldCondition(key="key", match=MatchValue(value=key)))
        return Filter(must=conditions)

    def put(self, namespace: Tuple[str, ...], key: str, value: Any):
        ns_string = "/".join(namespace)
        payload = {
            "namespace": ns_string,
            "key": key,
            "value": value
        }
        vector = self.embedding.embed_query(value if isinstance(value, str) else str(value))
        point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def get(self, namespace: Tuple[str, ...], key: str) -> Any:
        ns_string = "/".join(namespace)
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self._build_filter(ns_string, key),
            limit=1
        )[0]
        if results:
            return results[0].payload.get("value")
        return None

    def delete(self, namespace: Tuple[str, ...], key: str):
        ns_string = "/".join(namespace)
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self._build_filter(ns_string, key),
            limit=10
        )[0]
        ids = [pt.id for pt in results]
        if ids:
            self.client.delete(collection_name=self.collection_name, points_selector=ids)
    

    def search(
        self,
        namespace: Tuple[str, ...],
        query: str,
        *,
        limit: int = 3,
        filter: dict = None,
        offset: int = 0,
        **kwargs
    ) -> list[Document]:
        ns_string = "/".join(namespace)
        vector = self.embedding.embed_query(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=limit,
            offset=offset,
            query_filter=self._build_filter(ns_string),
            **kwargs
        )

        documents = []
        for hit in results:
            payload = hit.payload
            value = payload.get("value")

            if isinstance(value, dict):
                content = value.get("content", str(value))  # fallback if "content" missing
            else:
                content = str(value)

            documents.append(Document(
                page_content=content,
                metadata={"namespace": ns_string, "key": payload.get("key")},
                id=str(hit.id) if hit.id else None
            ))

        return documents

