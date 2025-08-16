import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        qdrant_path: str,
        collection_name: str = "documents",
        embedding_size: Optional[int] = None,
    ):
        """
        Initializes the Retriever with the embedding model, Qdrant path,
        collection name, and embedding size.

        Args:
            embedding_model (SentenceTransformer): The embedding model to use for
                encoding documents.
            qdrant_path (str): The path to the Qdrant database (local or URL).
            collection_name (str): The name of the collection in Qdrant to store
                documents.
            embedding_size (Optional[int]): The size of the embedding vectors. If
                None, it will be inferred from the model.
        
        Raises:
            ValueError: If embedding_size is not a positive integer.
        """
        self.embedding_model = embedding_model
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name

        # Connect to Qdrant local instance (using the local path)
        self.client = QdrantClient(
            path=self.qdrant_path,
        )

        # If embedding_size is not provided, attempt to infer it
        if embedding_size is None:
            embedding_size = self.embedding_model.get_sentence_embedding_dimension()
        else:
            if embedding_size <= 0:
                raise ValueError("Vector size must be a positive integer.")
        self.embedding_size = embedding_size

        # Ensure the collection exists
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
            )
    
    def add_documents(self, documents: list[str]) -> Dict[str, str]:
        """
        Embeds the documents and adds them to the Qdrant collection.
        Each document is assigned a unique UUID.
        
        Args:
            documents (list[str]): List of document texts to add.
            
        Returns:
            Dict[str, str]: Status and message of the operation.
        """
        try:
            embeddings = self.embedding_model.encode(documents)

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),  # unique id for each document
                    vector=embedding,
                    payload={"text": doc},
                )
                for doc, embedding in zip(documents, embeddings)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            return {"status": "success", "message": "Documents added successfully."}
        except Exception as e:
            logging.error(f"Error adding documents: {e}")
            return {"status": "error", "message": str(e)}

    def add_document(self, document: str) -> Dict[str, str]:
        """
        Embeds a single document and adds it to the Qdrant collection.
        The document is assigned a unique UUID.
        
        Args:
            document (str): The document text to add.
            
        Returns:
            Dict[str, str]: Status and message of the operation.
        """
        try:
            embedding = self.embedding_model.encode([document])[0]
            point = PointStruct(
                id=str(uuid.uuid4()),  # unique id for the document
                vector=embedding,
                payload={"text": document},
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])
            return {"status": "success", "message": "Document added successfully."}
        except Exception as e:
            logging.error(f"Error adding document: {e}")
            return {"status": "error", "message": str(e)}

    def retrieve(self, query: str, limit: int = 5) -> List[dict]:
        """
        Retrieves documents similar to the query using cosine similarity.
        Returns a list of dictionaries containing the document text and its score.
        
        Args:
            query (str): The search query.
            limit (int): Maximum number of results to return.
            
        Returns:
            List[dict]: List of dictionaries with 'text' and 'score' keys.
        """
        try:
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
            )

            results = [
                {"text": point.payload["text"], "score": point.score} 
                for point in search_result.points
            ]
            return results
        except Exception as e:
            logging.error(f"Error retrieving documents: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dict[str, Any]: Collection information including point count and config.
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.value,
                "status": collection_info.status.value
            }
        except Exception as e:
            logging.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def delete_document(self, document_id: str) -> Dict[str, str]:
        """
        Delete a document by its ID.
        
        Args:
            document_id (str): The ID of the document to delete.
            
        Returns:
            Dict[str, str]: Status and message of the operation.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[document_id]
            )
            return {"status": "success", "message": f"Document {document_id} deleted successfully."}
        except Exception as e:
            logging.error(f"Error deleting document: {e}")
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Test the Retriever class
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        "Alibaba-NLP/gte-multilingual-base", 
        trust_remote_code=True, 
        cache_folder="./cache"
    )
    
    print("Initializing Retriever...")
    retriever = Retriever(
        embedding_model=embedding_model,
        qdrant_path="./qdrant_database",
        collection_name="mcp_database",
    )
    
    # Get collection info
    print("\nCollection Information:")
    info = retriever.get_collection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test retrieval
    print("\nTesting retrieval with query: 'What is organic chemistry?'")
    results = retriever.retrieve("What is organic chemistry?", limit=5)
    
    if results:
        print(f"Found {len(results)} results:")
        print("-" * 80)
        for i, result in enumerate(results):
            text_preview = result['text'][:250] + "..." if len(result['text']) > 250 else result['text']
            print(f"Result {i + 1} (Score: {result['score']:.4f}):")
            print(f"Text: {text_preview}")
            print("-" * 80)
    else:
        print("No results found or error occurred.")
    
    # Test adding a new document
    print("\nTesting adding a new document...")
    test_doc = "This is a test document about machine learning and artificial intelligence."
    add_result = retriever.add_document(test_doc)
    print(f"Add document result: {add_result}")
    
    # Test retrieval with the new document
    print("\nTesting retrieval with query related to the new document...")
    ml_results = retriever.retrieve("machine learning", limit=3)
    if ml_results:
        print("Results for 'machine learning' query:")
        for i, result in enumerate(ml_results):
            text_preview = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
            print(f"{i + 1}. Score: {result['score']:.4f} - {text_preview}")
    
    print("\n✅ Retriever testing completed!")