import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client import models

# Định nghĩa các thư mục
DOCUMENT_DIR = "./documents"
QDRANT_DATABASE_PATH = "./qdrant_database"
CACHE_DIR = "./cache"

def discover_and_get_all_files(root_dir, allowed_extensions=None, recursive=False):
    """
    Tìm tất cả các file trong thư mục với các extension được phép
    """
    if allowed_extensions is None:
        allowed_extensions = ['.json', '.txt', '.md']

    all_files = []
    for item_name in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item_name)
        if os.path.isfile(item_path) and any(item_path.endswith(ext) for ext in allowed_extensions):
            all_files.append(item_path)
        elif os.path.isdir(item_path) and recursive:
            all_files.extend(discover_and_get_all_files(item_path, allowed_extensions, recursive))
    return all_files

def load_json_file(file_path):
    """Đọc file JSON và trả về dictionary"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_txt_file(file_path):
    """Đọc file text và trả về nội dung"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_md_file(file_path):
    """Đọc file markdown và trả về nội dung"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    print("Starting vector database creation process...")
    
    # 1. Tìm tất cả documents
    print("\n1. Discovering all documents...")
    all_document_paths = discover_and_get_all_files(
        DOCUMENT_DIR, 
        allowed_extensions=['.json', '.txt', '.md'], 
        recursive=True
    )
    print(f"Found {len(all_document_paths)} documents in {DOCUMENT_DIR}.")
    
    if len(all_document_paths) == 0:
        print("No documents found! Please run prepare_documents.py first.")
        return
    
    # 2. Load tất cả documents vào memory
    print("\n2. Loading all documents into memory...")
    documents = []
    with tqdm(total=len(all_document_paths), desc="Loading documents") as pbar:
        for doc_path in all_document_paths:
            try:
                if doc_path.endswith('.json'):
                    doc = load_json_file(doc_path)
                elif doc_path.endswith('.txt'):
                    doc = {"text": load_txt_file(doc_path)}
                elif doc_path.endswith('.md'):
                    doc = {"text": load_md_file(doc_path)}
                else:
                    continue

                # Add metadata
                doc["path"] = doc_path
                documents.append(doc)
                
            except Exception as e:
                print(f"Error loading {doc_path}: {str(e)}")
                continue
                
            pbar.update(1)
    
    print(f"Successfully loaded {len(documents)} documents.")
    
    if len(documents) == 0:
        print("No valid documents loaded! Exiting.")
        return
    
    # 3. Load embedding model
    print("\n3. Loading embedding model...")
    embedding_model_id = "Alibaba-NLP/gte-multilingual-base"
    try:
        embedding_model = SentenceTransformer(
            embedding_model_id,
            trust_remote_code=True,
            cache_folder=CACHE_DIR
        )
        print(f"Successfully loaded embedding model: {embedding_model_id}")
        print(f"Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")
    except Exception as e:
        print(f"Error loading embedding model: {str(e)}")
        return
    
    # 4. Convert documents to embeddings
    print("\n4. Converting documents to embeddings...")
    try:
        document_texts = [doc["text"] for doc in documents]
        document_embeddings = embedding_model.encode(
            document_texts,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=16,
        )
        print(f"Successfully created embeddings for {len(document_embeddings)} documents.")
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        return
    
    # 5. Create Qdrant client
    print("\n5. Connecting to Qdrant database...")
    try:
        # Tạo thư mục qdrant database nếu chưa tồn tại
        os.makedirs(QDRANT_DATABASE_PATH, exist_ok=True)
        
        # Sử dụng phương pháp lưu trực tiếp vào folder (phù hợp cho development)
        client = QdrantClient(path=QDRANT_DATABASE_PATH)
        print(f"Connected to Qdrant database at: {QDRANT_DATABASE_PATH}")
        
        # Nếu muốn sử dụng Docker container, uncomment dòng dưới và comment dòng trên:
        # client = QdrantClient(url="http://localhost:6333")
        # print("Connected to Qdrant database via Docker at: http://localhost:6333")
        
    except Exception as e:
        print(f"Error connecting to Qdrant database: {str(e)}")
        return
    
    # 6. Create collection
    print("\n6. Creating collection in Qdrant database...")
    collection_name = "mcp_database"
    try:
        # Kiểm tra xem collection đã tồn tại chưa
        collections = client.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
        
        if collection_exists:
            print(f"Collection '{collection_name}' already exists. Deleting it...")
            client.delete_collection(collection_name)
        
        # Tạo collection mới
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )
        print(f"Successfully created collection: {collection_name}")
        
    except Exception as e:
        print(f"Error creating collection: {str(e)}")
        return
    
    # 7. Upload points to Qdrant database
    print("\n7. Uploading documents and embeddings to Qdrant database...")
    try:
        points = []
        for idx, doc in enumerate(documents):
            point = models.PointStruct(
                id=idx,
                vector=document_embeddings[idx].tolist(),  # Convert tensor to list
                payload={
                    "text": doc["text"],
                    "path": doc["path"],
                    "source": doc.get("source", ""),
                    "split": doc.get("split", ""),
                    "id": doc.get("id", "")
                }
            )
            points.append(point)
        
        # Upload points in batches để tránh memory issues
        batch_size = 100
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        with tqdm(total=total_batches, desc="Uploading points") as pbar:
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                client.upload_points(
                    collection_name=collection_name,
                    points=batch
                )
                pbar.update(1)
        
        print(f"Successfully uploaded {len(points)} points to collection '{collection_name}'")
        
    except Exception as e:
        print(f"Error uploading points: {str(e)}")
        return
    
    # 8. Verify upload
    print("\n8. Verifying upload...")
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Collection info:")
        print(f"  - Name: {collection_info.config.params.vectors.size}")
        print(f"  - Vector size: {collection_info.config.params.vectors.size}")
        print(f"  - Distance metric: {collection_info.config.params.vectors.distance}")
        print(f"  - Points count: {collection_info.points_count}")
        
    except Exception as e:
        print(f"Error verifying upload: {str(e)}")
        return
    
    # 9. Test query để kiểm tra database hoạt động
    print("\n9. Testing database with example query...")
    try:
        test_query = "What is organic chemistry?"
        query_embedding = embedding_model.encode(test_query)
        
        hits = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=5,
        )
        
        print(f"\nExample query: '{test_query}'")
        print("Top 5 results:")
        print("-" * 80)
        
        for i, hit in enumerate(hits.points, 1):
            text_preview = hit.payload['text'][:250] + "..." if len(hit.payload['text']) > 250 else hit.payload['text']
            print(f"{i}. ID: {hit.id}")
            print(f"   Score: {hit.score:.4f}")
            print(f"   Source: {hit.payload.get('source', 'N/A')}")
            print(f"   Path: {hit.payload['path']}")
            print(f"   Text: {text_preview}")
            print("-" * 80)
        
        print("✅ Example query completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during test query: {str(e)}")
    
    print("\n🎉 Vector database creation completed successfully!")
    print(f"Database location: {QDRANT_DATABASE_PATH}")
    print(f"Collection name: {collection_name}")
    print(f"Total documents processed: {len(documents)}")

if __name__ == "__main__":
    main()