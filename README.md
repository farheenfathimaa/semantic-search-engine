# Semantic Search Engine for Research Papers

A powerful semantic search engine that allows users to search through research papers using natural language queries. Built with FastAPI, sentence transformers, and FAISS for efficient similarity search.

## Features

- 🔍 Semantic search using state-of-the-art embeddings
- ⚡ Fast similarity search with FAISS indexing
- 🌐 Modern web interface with responsive design
- 📊 Relevance scoring and ranking
- 🏷️ Keyword extraction and categorization
- 🔗 Direct links to original papers

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/farheenfathimaa/semantic-search-engine.git
   cd semantic-search-engine
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8000`

### Docker Deployment

1. **Build and run with Docker**
   ```bash
   docker-compose up --build
   ```

2. **Access the application**
   Open `http://localhost:8000` in your browser

## Project Structure

```
semantic-search-engine/
├── app.py                    # Main FastAPI application
├── data/
│   ├── embeddings.pkl       # Stored embeddings cache
│   └── sample_papers.json   # Sample research papers data
├── models/
│   └── embeddings.py        # Embedding model utilities
├── scripts/
│   └── run_and_export.py    # Data processing and export scripts
├── static/
│   ├── script.js           # Frontend JavaScript
│   └── style.css           # Styling
├── templates/
│   └── index.html          # Main web interface
├── tests/
│   └── test_search.py      # Unit tests
├── utils/
│   ├── __init__.py
│   ├── _pycache_/          # Python cache files
│   ├── data_processing.py   # Data processing utilities
│   ├── export.py           # Data export functionality
│   └── search.py           # Semantic search engine implementation
├── .env                    # Environment variables
├── .gitignore             # Git ignore rules
├── Dockerfile             # Docker configuration
├── README.md              # This file
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
└── search_results.csv     # Search results export
```

## API Endpoints

- `GET /` - Main search interface
- `POST /search` - Search with form data
- `GET /api/search?query=<query>&top_k=<number>` - JSON API endpoint

## Technologies Used

- **FastAPI** - Modern Python web framework
- **Sentence Transformers** - For creating semantic embeddings
- **FAISS** - For efficient similarity search
- **Docker** - For containerization
- **HTML/CSS/JavaScript** - For frontend interface
- **Jinja2** - For template rendering

## Configuration

The application uses environment variables for configuration. Create a `.env` file in the root directory:

```env
# Add your environment variables here
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
MAX_PAPERS=10000
EMBEDDING_DIM=384
```

## Adding More Papers

To add more papers to the search index:

1. **Update the data source** in `utils/data_processing.py`
2. **Modify the `load_sample_data()` function** to include your papers
3. **Delete `data/embeddings.pkl`** to force regeneration of embeddings
4. **Restart the application**

The system will automatically create new embeddings for the updated dataset.

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Deployment Options

### 1. Local Development
Follow the Quick Start guide above.

### 2. Cloud Deployment (Heroku/Railway/Render)
1. Push your code to GitHub
2. Connect to your preferred cloud platform
3. Set environment variables
4. Deploy

### 3. Docker Production
```bash
docker build -t semantic-search-engine .
docker run -p 8000:8000 semantic-search-engine
```

## Performance Notes

- **Initial startup**: Takes 1-2 minutes to create embeddings
- **Search queries**: Typically return results in <100ms
- **Memory usage**: ~2GB for 100,000 papers
- **Storage**: ~500MB for embeddings

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions, please open an issue on the GitHub repository.