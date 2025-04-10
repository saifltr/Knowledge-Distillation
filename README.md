# Document Pyramid Explorer

A Streamlit-based application for processing documents, building knowledge pyramids, and enabling conversational interactions with document content.

## Overview

Document Pyramid Explorer processes documents (PDF, Word, Excel, PowerPoint) into a hierarchical knowledge representation:

1. **Insights** - Atomic facts extracted from documents
2. **Concepts** - Higher-level themes connecting related information
3. **Abstracts** - Comprehensive document summaries
4. **Recollections** - Cross-document connections and insights

This knowledge pyramid enables efficient document understanding and intelligent question-answering about your documents.

## Features

- Document processing for multiple file formats (PDF, Word, Excel, PowerPoint)
- Automatic extraction of text and images with AI-generated descriptions
- Knowledge distillation into a hierarchical pyramid
- Vector-based search for efficient information retrieval
- Conversational interface for document querying

## Installation

### Prerequisites

- Python 3.8 or higher
- Mistral API key (a default one is provided but you can use your own)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/document-pyramid-explorer.git
   cd document-pyramid-explorer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create required directories:
   ```bash
   mkdir -p docs markdown_output markdown_output/images knowledge_pyramid knowledge_pyramid/insights knowledge_pyramid/concepts knowledge_pyramid/abstracts knowledge_pyramid/recollections knowledge_pyramid/vector_db
   ```

## Usage

### 1. Prepare Documents

Place your documents in the `docs` directory:

```bash
mkdir -p docs
# Copy your PDF, Word, Excel, or PowerPoint files into the docs directory
```

Supported file formats:
- PDF (.pdf)
- Microsoft Word (.doc, .docx)
- Microsoft Excel (.xls, .xlsx)
- Microsoft PowerPoint (.ppt, .pptx)

### 2. Run the Application

```bash
streamlit run app.py
```

This will start the Streamlit server and open the application in your web browser.

### 3. Using the Application

The application interface has three main tabs:

#### Document Status
- Shows the current status of document processing
- Displays previews of processed content

#### Knowledge Pyramid
- Explores the hierarchical knowledge representation
- Navigate through recollections, abstracts, concepts, and insights

#### Chat with Document
- Ask questions about your documents
- Receive AI-powered answers based on document content

### 4. Processing Flow

1. **Upload Document** - Upload your document through the Streamlit interface
2. **Process Document** - Extract text and images from the document
3. **Build Knowledge Pyramid** - Distill knowledge into hierarchical levels
4. **Explore** - Navigate the knowledge pyramid
5. **Chat** - Ask questions about your documents

## API Key

The application uses the Mistral AI API for document processing and interaction. A default API key is provided, but you can use your own by:

1. Setting the `MISTRAL_API_KEY` environment variable
2. Entering your API key in the sidebar input field

## Directory Structure

```
document-pyramid-explorer/
├── app.py                    # Main Streamlit application
├── file_processor.py         # Document processing functions
├── knowledge_distillation.py # Knowledge pyramid creation
├── query_pyramid.py          # Query processing against the pyramid
├── requirements.txt          # Python dependencies
├── docs/                     # Directory for input documents
├── markdown_output/          # Processed document content
│   └── images/               # Extracted images
└── knowledge_pyramid/        # Knowledge pyramid storage
    ├── insights/             # Atomic facts from documents
    ├── concepts/             # Higher-level themes
    ├── abstracts/            # Document summaries
    ├── recollections/        # Cross-document insights
    └── vector_db/            # Vector embeddings for search
```

## Example Workflow

1. Place a PDF document in the `docs` directory
2. Start the application with `streamlit run app.py`
3. Click "Process Document" to extract content
4. Click "Build Knowledge Pyramid" to create the knowledge structure
5. Explore the pyramid in the "Knowledge Pyramid" tab
6. Ask questions in the "Chat with Document" tab

## Limitations

- Large documents may take some time to process
- The quality of responses depends on the quality of the document content
- Very large images may cause memory issues
- Currently optimized for English language documents

## Technical Details

- Uses Mistral AI's OCR and language models for document processing
- Employs sentence transformers for vector embeddings
- Uses FAISS for efficient vector search
- Implements a hierarchical knowledge distillation process

