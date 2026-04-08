from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        self.dim = dim
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        if not ids:
            return

        if len(ids) != len(vectors) or len(ids) != len(payloads):
            raise ValueError(
                "Qdrant upsert requires ids, vectors, and payloads to have the same length"
            )

        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(collection_name=self.collection, points=points)


    def search(self, query_vector, top_k=5):
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        contexts = []
        sources = set()

        for r in response.points:
            payload = getattr(r, 'payload', None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text: 
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}