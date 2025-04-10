import os
import glob
import json
import faiss
import numpy as np
from tqdm import tqdm
from mistralai import Mistral
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Initialize Mistral client
api_key = "6iAQWtFQnCfR7kKaIrbhS3ZmlCFscPbd"
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

# Initialize sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use a larger model if desired

# Define paths
MARKDOWN_DIR = "markdown_output"
OUTPUT_DIR = "knowledge_pyramid"

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/insights", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/concepts", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/abstracts", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/recollections", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/vector_db", exist_ok=True)

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

# Function to extract insights from document pages
def extract_insights(document_path):
    print(f"Extracting insights from {document_path}")
    
    # Read the markdown document
    with open(document_path, 'r', encoding='utf-8') as f:
        document_content = f.read()
    
    # Split the document into pages (this is a simplification, you might need to adjust based on your markdown structure)
    pages = document_content.split('\n\n---\n\n')
    if len(pages) == 1:  # If no page markers, split by large paragraph breaks
        pages = document_content.split('\n\n\n')
    
    insights = []
    page_count = len(pages)
    
    # Process the document with a two-page sliding window
    all_insights = []
    
    for i in range(page_count - 1):
        two_page_content = pages[i] + "\n\n" + pages[i+1]
        
        # Prompt for insight extraction
        prompt = f"""You are an agent tasked with extracting atomic insights from documents.
        
I will provide you with content from two pages of a document. Your task is to extract atomic insights in simple subject-verb-object (SVO) format sentences.
Write sentences as if English is the second language of the user, focusing on clarity and precision.

Extract key facts, figures, relationships, and important information. Each insight should be a single, clear sentence.
Focus especially on extracting information from tables, converting tabular data into descriptive sentences.

Please return ONLY a numbered list of insights, with no additional text or explanation.

DOCUMENT CONTENT:
{two_page_content}

NUMBERED LIST OF INSIGHTS:"""
        
        response = get_llm_response(prompt)
        
        # Parse numbered list (assuming format like "1. Insight one\n2. Insight two")
        page_insights = []
        for line in response.strip().split('\n'):
            if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 100)):
                insight_text = line.split('.', 1)[1].strip()
                page_insights.append(insight_text)
        
        # Add insights to our collection
        all_insights.extend(page_insights)
        
    return all_insights

# Function to distill concepts from insights
def distill_concepts(insights):
    print("Distilling concepts from insights")
    
    # Join insights into a single text
    insights_text = "\n".join([f"- {insight}" for insight in insights])
    
    # Prompt for concept distillation
    prompt = f"""You are an agent tasked with identifying higher-level concepts from a list of atomic insights.
    
I will provide you with a list of detailed insights extracted from a document. Your task is to identify higher-level concepts that connect related information.
This step should significantly reduce noise and redundant information while preserving essential themes.

Please return ONLY a numbered list of concepts, with no additional text or explanation. Each concept should be a clear, concise statement.

INSIGHTS:
{insights_text}

NUMBERED LIST OF CONCEPTS:"""
    
    response = get_llm_response(prompt)
    
    # Parse numbered list
    concepts = []
    for line in response.strip().split('\n'):
        if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 100)):
            concept_text = line.split('.', 1)[1].strip()
            concepts.append(concept_text)
    
    return concepts

# Function to create abstract from concepts
def create_abstract(concepts, insights):
    print("Creating abstract from concepts")
    
    # Join concepts into a single text
    concepts_text = "\n".join([f"- {concept}" for concept in concepts])
    
    # Sample some insights (to avoid token limits)
    sampled_insights = insights[:min(50, len(insights))]
    insights_sample = "\n".join([f"- {insight}" for insight in sampled_insights])
    
    # Prompt for abstract creation
    prompt = f"""You are an agent tasked with writing a comprehensive abstract for a document.
    
I will provide you with a list of concepts and a sample of insights extracted from a document. Your task is to write an abstract that:
1. Is information-dense but clear
2. Captures the key themes and important information
3. Is more comprehensive than a typical human-written abstract

The abstract should be a coherent paragraph or two that summarizes the document effectively.

CONCEPTS:
{concepts_text}

SAMPLE INSIGHTS:
{insights_sample}

ABSTRACT:"""
    
    response = get_llm_response(prompt, max_tokens=1024)
    return response.strip()

# Function to store recollections across documents
def create_recollections(all_abstracts):
    print("Creating recollections across documents")
    
    # Join abstracts into a single text
    abstracts_text = "\n\n".join(all_abstracts)
    
    # Prompt for recollections creation
    prompt = f"""You are an agent tasked with creating high-level recollections across multiple documents.
    
I will provide you with abstracts from multiple documents. Your task is to identify critical information that is useful across all tasks.
These recollections should reveal relationships and information that are not apparent in a single document.

Please create a list of important recollections that would be useful for answering questions about this document collection.

ABSTRACTS:
{abstracts_text}

RECOLLECTIONS:"""
    
    response = get_llm_response(prompt, max_tokens=2048)
    return response.strip()

# Function to create embeddings and store in FAISS
def store_in_vector_db(levels_data):
    print("Storing in FAISS vector database")
    
    # For each level, create a separate vector index
    for level_name, items in levels_data.items():
        print(f"Processing {level_name}")
        
        texts = []
        metadata = []
        
        # Prepare texts and metadata
        for item in items:
            if isinstance(item, dict):
                texts.append(item['text'])
                metadata.append({k: v for k, v in item.items() if k != 'text'})
            else:
                texts.append(item)
                metadata.append({})
        
        # Create embeddings
        embeddings = embedding_model.encode(texts)
        
        # Create a FAISS index
        vector_dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(vector_dimension)
        
        # Add vectors to the index
        index.add(np.array(embeddings).astype('float32'))
        
        # Save the index
        faiss.write_index(index, f"{OUTPUT_DIR}/vector_db/{level_name}_index.faiss")
        
        # Save the texts and metadata
        with open(f"{OUTPUT_DIR}/vector_db/{level_name}_data.json", 'w', encoding='utf-8') as f:
            json.dump([{"text": text, "metadata": meta} for text, meta in zip(texts, metadata)], f, indent=2)

# Main function to process all documents
def process_documents():
    # Get all markdown files
    markdown_files = glob.glob(f"{MARKDOWN_DIR}/**/*.md", recursive=True)
    
    all_insights = []
    all_concepts = []
    all_abstracts = []
    
    # Process each document
    for doc_path in tqdm(markdown_files):
        doc_name = Path(doc_path).stem
        
        # Extract insights
        insights = extract_insights(doc_path)
        all_insights.extend([{"text": insight, "source": doc_name} for insight in insights])
        
        # Save insights
        with open(f"{OUTPUT_DIR}/insights/{doc_name}_insights.json", 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2)
        
        # Distill concepts
        concepts = distill_concepts(insights)
        all_concepts.extend([{"text": concept, "source": doc_name} for concept in concepts])
        
        # Save concepts
        with open(f"{OUTPUT_DIR}/concepts/{doc_name}_concepts.json", 'w', encoding='utf-8') as f:
            json.dump(concepts, f, indent=2)
        
        # Create abstract
        abstract = create_abstract(concepts, insights)
        all_abstracts.append({"text": abstract, "source": doc_name})
        
        # Save abstract
        with open(f"{OUTPUT_DIR}/abstracts/{doc_name}_abstract.txt", 'w', encoding='utf-8') as f:
            f.write(abstract)
    
    # Create recollections
    recollections = create_recollections([abstract["text"] for abstract in all_abstracts])
    
    # Save recollections
    with open(f"{OUTPUT_DIR}/recollections/recollections.txt", 'w', encoding='utf-8') as f:
        f.write(recollections)
    
    # Store in vector database
    levels_data = {
        "insights": all_insights,
        "concepts": all_concepts,
        "abstracts": all_abstracts,
        "recollections": [{"text": recollections, "source": "all_documents"}]
    }
    
    store_in_vector_db(levels_data)
    
    print(f"Knowledge pyramid processing complete. Data stored in {OUTPUT_DIR}")

if __name__ == "__main__":
    process_documents()