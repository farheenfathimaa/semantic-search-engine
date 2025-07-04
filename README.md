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

### Option 1: Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Open http://localhost:8000 in your browser

### Option 2: Docker

1. Build and run with Docker:
   ```bash
   docker-compose up --build
   ```
2. Access the application at http://localhost:8000

## Project Structure

- `app.py` - Main FastAPI application
- `utils/search.py` - Semantic search engine implementation
- `utils/data_processing.py` - Data processing utilities
- `templates/index.html` - Frontend HTML template
- `static/` - CSS and JavaScript files
- `data/` - Sample papers and embeddings storage

## API Endpoints

- `GET /` - Main search interface
- `POST /search` - Search with form data
- `GET /api/search?query=<query>&top_k=<number>` - JSON API endpoint

## Technologies Used

- **FastAPI** - Modern web framework
- **Sentence Transformers** - For creating embeddings
- **FAISS** - For efficient similarity search
- **Docker** - For containerization
- **HTML/CSS/JavaScript** - For frontend

## Adding More Papers

To add more papers to the search index:

1. Modify `utils/data_processing.py`
2. Add papers to the `create_sample_papers()` method
3. Delete `data/embeddings.pkl` to regenerate embeddings
4. Restart the application

## Deployment

### GitHub Pages (Frontend Only)
For a static version, you can deploy the frontend to GitHub Pages.

### Heroku/Railway/Render
For full deployment with backend:

1. Push to GitHub
2. Connect to your preferred platform
3. Set environment variables
4. Deploy

## Performance Notes

- Initial startup takes 1-2 minutes to create embeddings
- Search queries typically return results in <100ms
- Memory usage: ~2GB for 100,000 papers
- Storage: ~500MB for embeddings

## License

MIT License