# Personal Database MCP Server

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-compatible-orange.svg)](https://modelcontextprotocol.io/)

A powerful **Model Context Protocol (MCP) server** that provides intelligent document retrieval and search capabilities using vector embeddings and semantic similarity. This server combines local document storage with internet search functionality to create a comprehensive knowledge base for AI assistants.

## 🚀 Features

### Core Functionality
- **Vector Database Storage**: Efficient document storage using Qdrant vector database
- **Semantic Search**: Advanced similarity search using multilingual embeddings
- **Internet Search Integration**: Fallback to DuckDuckGo search when local documents are insufficient
- **Dynamic Document Addition**: Add new documents to the database on-the-fly
- **Topic Organization**: Hierarchical document organization by topics/categories

### MCP Protocol Support
- **Tools**: Document retrieval, internet search, and document addition
- **Resources**: Browse documents by topics with pagination support
- **Prompts**: Pre-configured prompts for various retrieval scenarios

### Technical Features
- **Multilingual Support**: Using Alibaba's GTE multilingual embedding model
- **Scalable Architecture**: Batch processing and efficient memory management
- **Real-time Updates**: Live document addition without server restart
- **Flexible File Formats**: Support for JSON, TXT, and Markdown files

## 📋 Prerequisites

- Python 3.11 or higher
- 8GB+ RAM (recommended for embedding model)
- 2GB+ free disk space for vector database

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/personal-database-mcp-server.git
   cd personal-database-mcp-server
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

## 🚀 Quick Start

### 1. Prepare Documents
First, download and prepare the educational datasets:

```bash
python prepare_documents.py
```

This will download 24 educational datasets covering various subjects like:
- Science (Physics, Chemistry, Biology)
- Social Studies (History, Philosophy, Religion)
- Health Sciences (Medicine, Psychology, Psychiatry)
- Arts & Culture
- Education (Pedagogy)

### 2. Create Vector Database
Build the vector database from your documents:

```bash
python create_vector_database.py
```

This process will:
- Load all documents from the `./documents` folder
- Generate embeddings using the multilingual model
- Store vectors in Qdrant database
- Create searchable indices

### 3. Start MCP Server
Launch the MCP server:

```bash
python server.py
```

The server will start on `http://127.0.0.1:2545` and provide MCP-compatible endpoints.

### 4. Test the Server
Test document retrieval:

```bash
python retriever.py
```

## 🔧 Configuration

### Directory Structure
```
personal_database_mcp_server/
├── .venv/                    # Virtual environment
├── documents/                # Document storage by topics
│   ├── chemistry_textbook/   # Topic-based folders
│   ├── physics_wiki/
│   └── ...
├── qdrant_database/         # Vector database storage
├── cache/                   # Model cache
├── create_vector_database.py # Database creation script
├── prepare_documents.py     # Dataset preparation
├── retriever.py            # Retriever class
├── server.py               # MCP server implementation
└── README.md
```

### Environment Variables
```bash
# Optional: Custom paths
export DOCUMENT_DIR="./documents"
export QDRANT_DATABASE_PATH="./qdrant_database"
export CACHE_DIR="./cache"
```

## 📚 Usage Examples

### MCP Tools

#### 1. Retrieve Documents from Database
```python
# Query: "What is organic chemistry?"
# Returns: Top 5 most similar documents with scores
```

#### 2. Search Internet
```python
# Query: "Latest AI research 2024"
# Returns: Recent search results from DuckDuckGo
```

#### 3. Add Document to Database
```python
# Add new document with optional topic classification
# Automatically indexes for future retrieval
```

### MCP Resources

#### Browse Topics
```
GET document://topics
# Returns: List of all available topics
```

#### Get Documents by Topic
```
GET document://topics/chemistry_textbook
# Returns: All documents in chemistry textbook category
```

#### Paginated Access
```
GET document://topics/physics_wiki/pages/1
# Returns: First 10 documents from physics wiki
```

### MCP Prompts

#### Database Retrieval Prompt
Optimized prompt for retrieving relevant documents from the local database.

#### Hybrid Search Prompt
Combines local database search with internet search for comprehensive results.

#### Internet-Only Search Prompt
Direct internet search when local knowledge is insufficient.

## 🔍 API Reference

### Tools

| Tool Name | Description | Parameters |
|-----------|-------------|------------|
| `retrieve_documents_from_database` | Search local vector database | `query: str, num_documents: int` |
| `search_query_on_internet` | Search using DuckDuckGo | `query: str, num_documents: int` |
| `add_document_to_database` | Add new document | `document: str, topic_name?: str, document_name?: str` |

### Resources

| Resource URI | Description |
|--------------|-------------|
| `document://topics` | Get all available topics |
| `document://topics/{topic_name}` | Get all documents by topic |
| `document://topics/{topic_name}/pages/{page_number}` | Paginated topic access |

### Response Schemas

```python
class RetrievedDocument(BaseModel):
    text: str
    score: Optional[float]

class RetrievalResult(BaseModel):
    results: List[RetrievedDocument]

class AddDocumentResponse(BaseModel):
    status: str
    message: str
```

## 🧪 Testing

Run the test suite:

```bash
# Test retriever functionality
python retriever.py

# Test vector database creation
python create_vector_database.py

# Test MCP server endpoints
python server.py --test
```

## 📊 Performance

### Benchmarks
- **Document Retrieval**: <100ms for typical queries
- **Embedding Generation**: ~50ms per document
- **Database Creation**: ~2-5 minutes for 10K documents
- **Memory Usage**: ~2GB with loaded embedding model

### Optimization Tips
- Use SSD storage for better I/O performance
- Increase batch size for bulk operations
- Monitor RAM usage during large dataset processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Format code
black .
isort .

# Type checking
mypy .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/) for the MCP specification
- [Qdrant](https://qdrant.tech/) for the vector database
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [Alibaba DAMO Academy](https://github.com/FlagOpen/FlagEmbedding) for GTE multilingual embeddings
- [Hugging Face](https://huggingface.co/) for dataset hosting

## 📞 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: [GitHub Issues](https://github.com/yourusername/personal-database-mcp-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/personal-database-mcp-server/discussions)

## 🗺️ Roadmap

### v0.2.0
- [ ] Support for additional file formats (PDF, DOCX)
- [ ] Advanced filtering and faceted search
- [ ] Multi-language query translation
- [ ] Performance monitoring dashboard

### v0.3.0
- [ ] Distributed vector database support
- [ ] Advanced document preprocessing
- [ ] Custom embedding model fine-tuning
- [ ] RESTful API endpoints

### v1.0.0
- [ ] Production-ready deployment scripts
- [ ] Enterprise security features
- [ ] Advanced analytics and reporting
- [ ] Plugin system for extensibility
