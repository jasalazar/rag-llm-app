# Rag-LLM-App
Previously, I built an agentic application using LangChain, made for the UI: https://github.com/jasalazar/basic-agentic-app

Now, I built an LLM application with LangChain, FastAPI, Rag pipelines using multi-representation indexing, traceability and monitoring using LangSmith, and VoyageAI's voyage-3 as embedding model.

## Assumptions
-You already have Python installed  
-You have an Anthropic API key with credits loaded  
-You have a LangSmith API key  
-You have a VoyageAI API key 

## First step
Install libraries and dependencies `pip install -r requirements.txt`

## Second step - Run the application
-On a CLI run `uvicorn main:app --reload --port 8000`  
-Open a browser and go to `http://127.0.0.1:8000/` 
