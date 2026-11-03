import os
from dotenv import load_dotenv

from langsmith import traceable
from configs.config import Settings

settings = Settings()

load_dotenv()


model = settings.rerank_model
top_n = settings.rerank_top_n

threshold = settings.rerank_threshold

@traceable(name="cohere_rerank", run_type="retriever")
def rerank_documents(co, query, documents):
    response = co.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=top_n,
    )

    # --- INJECT THIS DEBUG BLOCK ---
    print(f"\n--- RERANKER DEBUG ---")
    print(f"Threshold Set To: {threshold} (Type: {type(threshold)})")
    print(f"Query sent to Cohere: '{query}'")
    for r in response.results:
        print(f"Chunk {r.index} Score: {r.relevance_score}")
    print(f"----------------------\n")

    # Filter out chunks that do not meet the minimum relevance threshold
    filtered_results = [
        r for r in response.results 
        if r.relevance_score >= threshold
    ]
    
    # Overwrite the response results with the filtered list
    response.results = filtered_results

    return response