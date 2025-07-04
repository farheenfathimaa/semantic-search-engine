# app.py
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from dotenv import load_dotenv
from utils.search import SemanticSearchEngine
from utils.data_processing import DataProcessor

# Load environment variables
load_dotenv()

# (OPTIONAL) If you still want to keep model download logic, 
# make sure it's compatible or comment it out.
# from huggingface_hub import HfApi
# def download_model(model_name):
#     api = HfApi()
#     model_info = api.model_info(model_name)
#     return model_info

# model_name = os.getenv("MODEL_NAME")
# if model_name:
#     download_model(model_name)

app = FastAPI(title="Semantic Search Engine for Research Papers")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize search engine
search_engine = None

@app.on_event("startup")
async def startup_event():
    global search_engine
    search_engine = SemanticSearchEngine()

    # Load or create embeddings
    if not os.path.exists("data/embeddings.pkl"):
        print("Creating embeddings for the first time...")
        data_processor = DataProcessor()
        papers = data_processor.load_sample_data()
        search_engine.create_embeddings(papers)
        print("Embeddings created successfully!")
    else:
        search_engine.load_embeddings()
        print("Embeddings loaded successfully!")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search")
async def search(request: Request, query: str = Form(...)):
    try:
        results = search_engine.search(query, top_k=10)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "query": query,
            "results": results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
async def api_search(query: str, top_k: int = 10):
    try:
        results = search_engine.search(query, top_k=top_k)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
