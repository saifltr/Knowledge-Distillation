import os
import json
import faiss
import numpy as np
from mistralai import Mistral
from sentence_transformers import SentenceTransformer

# Initialize Mistral client
api_key = "6iAQWtFQnCfR7kKaIrbhS3ZmlCFscPbd"
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

# Initialize sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define paths
OUTPUT_DIR = "knowledge_pyramid"
VECTOR_DB_DIR = f"{OUTPUT_DIR}/vector_db"

# Function to perform LLM inference
def get_llm_response(prompt, max_tokens=4096):
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=max_tokens
    )
    return chat_response.choices[0].message.content

# Function to search in vector database
def search_vector_db(query, level_name, top_k=5):
    # Create query embedding
    query_embedding = embedding_model.encode([query])
    
    # Load the index
    index = faiss.read_index(f"{VECTOR_DB_DIR}/{level_name}_index.faiss")
    
    # Search
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    
    # Load the data
    with open(f"{VECTOR_DB_DIR}/{level_name}_data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get the results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(data):
            results.append({
                "text": data[idx]["text"],
                "metadata": data[idx].get("metadata", {}),
                "distance": float(distances[0][i])
            })
    
    return results

# Function to process a user query using the pyramid approach
def process_query(query):
    print(f"Processing query: {query}")
    
    # First, generate search terms from the query
    prompt = f"""You are an agent tasked with generating search terms from a user query.
    
USER QUERY:
{query}

Generate 3-5 search terms that would help find relevant information in our knowledge base.
Each search term should be a concise phrase or concept related to the query.

SEARCH TERMS:"""
    
    search_terms_response = get_llm_response(prompt)
    search_terms = [term.strip() for term in search_terms_response.strip().split('\n') if term.strip()]
    
    print(f"Generated search terms: {search_terms}")
    
    # Initialize collections for search results
    recollection_results = []
    abstract_results = []
    concept_results = []
    insight_results = []
    
    # Search each level of the pyramid
    for term in search_terms:
        # Start with recollections (highest level)
        recollection_results.extend(search_vector_db(term, "recollections", top_k=2))
        
        # Search abstracts
        abstract_results.extend(search_vector_db(term, "abstracts", top_k=3))
        
        # Search concepts
        concept_results.extend(search_vector_db(term, "concepts", top_k=5))
        
        # Search insights (lowest level)
        insight_results.extend(search_vector_db(term, "insights", top_k=10))
    
    # Remove duplicates by text
    def deduplicate(results):
        seen = set()
        deduplicated = []
        for result in results:
            if result["text"] not in seen:
                seen.add(result["text"])
                deduplicated.append(result)
        return deduplicated
    
    recollection_results = deduplicate(recollection_results)
    abstract_results = deduplicate(abstract_results)
    concept_results = deduplicate(concept_results)
    insight_results = deduplicate(insight_results)
    
    # Sort by distance
    recollection_results.sort(key=lambda x: x["distance"])
    abstract_results.sort(key=lambda x: x["distance"])
    concept_results.sort(key=lambda x: x["distance"])
    insight_results.sort(key=lambda x: x["distance"])
    
    # Limit number of results to avoid token overflow
    recollection_results = recollection_results[:2]
    abstract_results = abstract_results[:3]
    concept_results = concept_results[:5]
    insight_results = insight_results[:10]
    
    # Prepare context for LLM
    context = ""
    
    if recollection_results:
        context += "RECOLLECTIONS:\n"
        for i, result in enumerate(recollection_results):
            context += f"{i+1}. {result['text']}\n\n"
    
    if abstract_results:
        context += "ABSTRACTS:\n"
        for i, result in enumerate(abstract_results):
            source = result.get("metadata", {}).get("source", "unknown")
            context += f"{i+1}. [Source: {source}] {result['text']}\n\n"
    
    if concept_results:
        context += "CONCEPTS:\n"
        for i, result in enumerate(concept_results):
            source = result.get("metadata", {}).get("source", "unknown")
            context += f"{i+1}. [Source: {source}] {result['text']}\n\n"
    
    if insight_results:
        context += "INSIGHTS:\n"
        for i, result in enumerate(insight_results):
            source = result.get("metadata", {}).get("source", "unknown")
            context += f"{i+1}. [Source: {source}] {result['text']}\n\n"
    
    # Generate response
    prompt = f"""You are an assistant with access to a knowledge base organized in a pyramid structure.
I will provide you with retrieved information from various levels of the pyramid:
- RECOLLECTIONS: High-level information useful across all tasks
- ABSTRACTS: Comprehensive summaries of documents
- CONCEPTS: Higher-level concepts that connect related information
- INSIGHTS: Atomic facts extracted from documents

Using this information, please provide a comprehensive answer to the user's query.
Cite sources when appropriate using [Source: document_name] format.

USER QUERY:
{query}

RETRIEVED INFORMATION:
{context}

ANSWER:"""
    
    response = get_llm_response(prompt, max_tokens=2048)
    return response

# Main function to handle user queries
def main():
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        response = process_query(query)
        print("\nResponse:")
        print(response)

if __name__ == "__main__":
    main()