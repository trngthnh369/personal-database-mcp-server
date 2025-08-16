import logging
import os
from typing import Any, Dict, List, Optional
import uuid
import json

import click
from langchain_community.tools import DuckDuckGoSearchResults
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from retriever import Retriever
from create_vector_database import (
    discover_and_get_all_files,
    load_json_file,
    load_txt_file,
    load_md_file,
)

# ===== PYDANTIC MODELS =====

# Input schemas
class QueryInput(BaseModel):
    query: str = Field(description="The query to retrieve relevant documents.")
    num_documents: int = Field(default=5, description="The number of documents to retrieve for the query.")

# Output schemas for tools
class RetrievedDocument(BaseModel):
    text: str = Field(description="The content of the retrieved document.")
    score: Optional[float] = Field(None, description="The score of the retrieved document based on similarity.")

class RetrievalResult(BaseModel):
    results: List[RetrievedDocument] = Field(description="List of retrieved documents with their scores.")

class SearchResult(BaseModel):
    results: List[RetrievedDocument] = Field(description="List of search results from the internet.")

class AddDocumentResponse(BaseModel):
    status: str = Field(description="Status of the operation.")
    message: str = Field(description="Message describing the result.")

# Output schemas for resources
class LocalDocument(BaseModel):
    text: str = Field(description="The content of the local document.")
    path: str = Field(description="The path of the local document.")
    source: Optional[str] = Field(None, description="The source of the local document.")
    split: Optional[str] = Field(None, description="The split of the local document.")
    id: Optional[str] = Field(None, description="The id of the local document.")

class LocalDocumentList(BaseModel):
    documents: List[LocalDocument] = Field(description="The list of local documents.")


def create_mcp_server(
    retriever: Retriever,
    documents_path_by_topics: Dict[str, List[str]],
    all_document_paths: List[str],
):
    """Create and configure the MCP server with tools, resources, and prompts."""
    
    mcp = FastMCP(
        name="Retriever MCP Server",
        host="127.0.0.1",
        port=2545,
        description="A server that provides MCP-powered agentic RAG capabilities.",
    )

    # ===== TOOLS =====

    @mcp.tool(
        name="retrieve_documents_from_database",
        title="Retrieve Documents from Database",
        description="Retrieve documents similar to the query from the database.",
    )
    def retrieve(input: QueryInput) -> RetrievalResult:
        """Retrieve documents from the vector database based on similarity to the query."""
        retrieval_results = retriever.retrieve(input.query, input.num_documents)
        return RetrievalResult(
            results=[
                RetrievedDocument(text=result["text"], score=result["score"])
                for result in retrieval_results
            ]
        )

    @mcp.tool(
        name="search_query_on_internet",
        title="Search the Query on the Internet",
        description="Search for documents on the internet related to the query. This tool will be used when the database does not have relevant documents for the query.",
    )
    def search_query_on_internet(input: QueryInput) -> SearchResult:
        """Search for information on the internet using DuckDuckGo."""
        search_engine = DuckDuckGoSearchResults(
            output_format="list", num_results=input.num_documents, backend="text"
        )
        search_results = search_engine.invoke(input.query)
        return SearchResult(
            results=[
                RetrievedDocument(
                    text=f"Title: {result['title']}\nText: {result['snippet']}",
                    score=None,
                )
                for result in search_results
            ]
        )

    @mcp.tool(
        name="add_document_to_database",
        title="Add a Document to Database",
        description="Add a document to the database for future retrieval.",
    )
    def add_document_to_database(
        document: str, 
        topic_name: Optional[str] = None, 
        document_name: Optional[str] = None
    ) -> AddDocumentResponse:
        """Add a document to both the file system and vector database."""
        doc_id = str(uuid.uuid4())
        
        if topic_name is None:
            topic_name = "default"
        if not os.path.exists(f"./documents/{topic_name}"):
            os.makedirs(f"./documents/{topic_name}")
        if document_name is None:
            document_name = doc_id

        # Save to the documents folder
        doc_path = f"./documents/{topic_name}/{document_name}.json"
        doc_data = {
            "text": document,
            "path": doc_path,
            "source": "user-added",
            "split": topic_name,
            "id": doc_id,
        }
        
        with open(doc_path, "w", encoding="utf-8") as f:
            json.dump(doc_data, f, indent=4, ensure_ascii=False)

        # Add to the vector database
        add_result = retriever.add_document(document)
        
        if add_result["status"] == "success":
            return AddDocumentResponse(status="success", message="Document added to database.")
        else:
            return AddDocumentResponse(status="error", message=f"Error adding document: {add_result['message']}")

    # ===== RESOURCES =====

    @mcp.resource(
        uri="document://topics",
        name="get_all_topics",
        title="Get All Topics in the Database",
        description="A resource to get all topics in the database.",
    )
    def get_all_topics() -> List[str]:
        """Get all available topics in the database."""
        return sorted(list(documents_path_by_topics.keys()))

    @mcp.resource(
        uri="document://topics/{topic_name}",
        name="get_all_documents_by_topic",
        title="Get All Documents by Topic",
        description="A resource to get all documents by topic.",
    )
    def get_all_documents_by_topic(topic_name: str) -> LocalDocumentList:
        """Get all documents for a specific topic."""
        if topic_name not in documents_path_by_topics:
            return LocalDocumentList(documents=[])
            
        doc_paths = documents_path_by_topics[topic_name]
        documents = []
        
        for doc_path in doc_paths:
            try:
                if doc_path.endswith(".json"):
                    doc_data = load_json_file(doc_path)
                    doc = LocalDocument(
                        text=doc_data["text"],
                        path=doc_path,
                        source=doc_data.get("source", ""),
                        split=doc_data.get("split", ""),
                        id=doc_data.get("id", ""),
                    )
                elif doc_path.endswith(".txt"):
                    doc = LocalDocument(
                        text=load_txt_file(doc_path),
                        path=doc_path,
                        source=None,
                        split=None,
                        id=None,
                    )
                elif doc_path.endswith(".md"):
                    doc = LocalDocument(
                        text=load_md_file(doc_path),
                        path=doc_path,
                        source=None,
                        split=None,
                        id=None,
                    )
                else:
                    continue
                documents.append(doc)
            except Exception as e:
                logging.error(f"Error loading document {doc_path}: {e}")
                continue
                
        return LocalDocumentList(documents=documents)

    @mcp.resource(
        uri="document://topics/{topic_name}/pages/{page_number}",
        name="get_documents_by_topic",
        title="Get Documents by Topic",
        description="A resource to get documents by topic using pagination. Each page contains 10 documents.",
    )
    def get_documents_by_topic(topic_name: str, page_number: int) -> LocalDocumentList:
        """Get documents for a specific topic with pagination."""
        if topic_name not in documents_path_by_topics:
            return LocalDocumentList(documents=[])
            
        doc_paths = documents_path_by_topics[topic_name]
        start_index, end_index = (page_number - 1) * 10, page_number * 10
        documents = []
        
        for doc_path in doc_paths[start_index:end_index]:
            try:
                if doc_path.endswith(".json"):
                    doc_data = load_json_file(doc_path)
                    doc = LocalDocument(
                        text=doc_data["text"],
                        path=doc_path,
                        source=doc_data.get("source", ""),
                        split=doc_data.get("split", ""),
                        id=doc_data.get("id", ""),
                    )
                elif doc_path.endswith(".txt"):
                    doc = LocalDocument(
                        text=load_txt_file(doc_path),
                        path=doc_path,
                        source=None,
                        split=None,
                        id=None,
                    )
                elif doc_path.endswith(".md"):
                    doc = LocalDocument(
                        text=load_md_file(doc_path),
                        path=doc_path,
                        source=None,
                        split=None,
                        id=None,
                    )
                else:
                    continue
                documents.append(doc)
            except Exception as e:
                logging.error(f"Error loading document {doc_path}: {e}")
                continue
                
        return LocalDocumentList(documents=documents)

    # ===== PROMPTS =====

    @mcp.prompt(
        name="retrieve_documents_from_database_prompt",
        title="Retrieve Documents from Database Prompt",
        description="Prompt to retrieve documents similar to the query from the database.",
    )
    def get_retrieve_document_from_database_tool(input: QueryInput) -> base.UserMessage:
        """Generate a prompt for retrieving documents from the database."""
        return base.UserMessage(
            content=f"""Answer the following question. Remember to use the 'retrieve_documents_from_database' tool to find {input.num_documents} relevant documents in the database to support your answer.

Question: {input.query}
"""
        )

    @mcp.prompt(
        name="retrieve_document_and_search_internet_prompt",
        title="Retrieve Document and Search Internet Prompt",
        description="Prompt to retrieve documents similar to the query from the database and search the internet if necessary.",
    )
    def get_retrieve_document_and_search_internet_tool(input: QueryInput) -> base.UserMessage:
        """Generate a prompt for retrieving documents from database and searching internet if needed."""
        return base.UserMessage(
            content=f"""Answer the following question. Remember to use the 'retrieve_documents_from_database' tool to find {input.num_documents} relevant documents in the database to support your answer. If the database does not have relevant documents, use the 'search_query_on_internet' tool to search for documents on the internet.

Question: {input.query}
"""
        )

    @mcp.prompt(
        name="search_query_on_internet_prompt",
        title="Search Internet Prompt",
        description="Prompt to search the internet for documents related to the query.",
    )
    def get_search_internet_tool(input: QueryInput) -> base.UserMessage:
        """Generate a prompt for searching the internet directly."""
        return base.UserMessage(
            content=f"""Answer the following question. Remember to use the 'search_query_on_internet' tool to find {input.num_documents} relevant documents on the internet to support your answer.

Question: {input.query}
"""
        )

    @mcp.prompt(
        name="add_single_document_to_database_prompt",
        title="Add Single Document to Database Prompt",
        description="Prompt to add a single document to the database.",
    )
    def get_add_single_document_to_database_tool(document: str) -> base.UserMessage:
        """Generate a prompt for adding a document to the database."""
        return base.UserMessage(
            content=f"""Add the following document to the database for future retrieval using the 'add_document_to_database' tool.

Document: {document}
"""
        )

    return mcp


def run_mcp_server():
    """Initialize and run the Personal Database MCP Server."""
    print("🚀 Starting Personal Database MCP Server...")
    
    # Initialize embedding model
    print("📚 Loading embedding model...")
    embedding_model = SentenceTransformer(
        "Alibaba-NLP/gte-multilingual-base",
        trust_remote_code=True,
        cache_folder="./cache",
    )
    print("✅ Embedding model loaded successfully!")
    
    # Initialize retriever
    print("🔍 Initializing retriever...")
    qdrant_path = "./qdrant_database"
    retriever = Retriever(
        embedding_model=embedding_model,
        qdrant_path=qdrant_path,
        collection_name="mcp_database",
    )
    print("✅ Retriever initialized successfully!")
    
    # Load document paths by topics
    print("📁 Loading document paths...")
    DOCUMENT_DIR = "./documents"
    DOCUMENTS_PATH_BY_TOPICS = {}
    
    if os.path.exists(DOCUMENT_DIR):
        for folder in os.listdir(DOCUMENT_DIR):
            folder_path = os.path.join(DOCUMENT_DIR, folder)
            if os.path.isdir(folder_path):
                DOCUMENTS_PATH_BY_TOPICS[folder] = discover_and_get_all_files(
                    folder_path,
                    allowed_extensions=[".json", ".txt", ".md"],
                    recursive=True,
                )
        
        ALL_DOCUMENT_PATHS = []
        for folder in DOCUMENTS_PATH_BY_TOPICS:
            ALL_DOCUMENT_PATHS.extend(DOCUMENTS_PATH_BY_TOPICS[folder])
        
        print(f"📊 Found {len(DOCUMENTS_PATH_BY_TOPICS)} topics with {len(ALL_DOCUMENT_PATHS)} total documents")
        for topic, paths in DOCUMENTS_PATH_BY_TOPICS.items():
            print(f"  📂 {topic}: {len(paths)} documents")
    else:
        print(f"⚠️  Documents directory not found: {DOCUMENT_DIR}")
        DOCUMENTS_PATH_BY_TOPICS = {}
        ALL_DOCUMENT_PATHS = []
    
    # Create and run MCP server
    print("🏗️  Creating MCP server...")
    mcp_server = create_mcp_server(retriever, DOCUMENTS_PATH_BY_TOPICS, ALL_DOCUMENT_PATHS)
    
    print("🌐 Starting MCP server on http://127.0.0.1:2545")
    print("🎯 Available tools:")
    print("  • retrieve_documents_from_database")
    print("  • search_query_on_internet") 
    print("  • add_document_to_database")
    print("🎯 Available resources:")
    print("  • document://topics")
    print("  • document://topics/{topic_name}")
    print("  • document://topics/{topic_name}/pages/{page_number}")
    print("🎯 Available prompts:")
    print("  • retrieve_documents_from_database_prompt")
    print("  • retrieve_document_and_search_internet_prompt")
    print("  • search_query_on_internet_prompt")
    print("  • add_single_document_to_database_prompt")
    print("\n✨ Server ready! Press Ctrl+C to stop.")
    
    mcp_server.run(transport="streamable-http")


@click.command()
def main():
    """Main entry point for the MCP server."""
    try:
        run_mcp_server()
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except Exception as e:
        print(f"❌ Error running server: {e}")
        logging.error(f"Server error: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()