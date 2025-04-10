import os
import sys
import time
import streamlit as st
import subprocess
import pandas as pd
import json
from pathlib import Path
import base64
import file_processor
import knowledge_distillation
import query_pyramid
from mistralai import Mistral
import tempfile
import shutil

# Set page configuration
st.set_page_config(
    page_title="Document Pyramid Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: 600;
    }
    .section {
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .pyramid-level {
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    .pyramid-recollections {
        background-color: #bbdefb;
    }
    .pyramid-abstracts {
        background-color: #c8e6c9;
    }
    .pyramid-concepts {
        background-color: #fff9c4;
    }
    .pyramid-insights {
        background-color: #ffccbc;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 5px solid #757575;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'pyramid_built' not in st.session_state:
    st.session_state.pyramid_built = False
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# API key setup
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.environ.get("MISTRAL_API_KEY", "6iAQWtFQnCfR7kKaIrbhS3ZmlCFscPbd")

# Get Mistral client
@st.cache_resource
def get_mistral_client():
    return Mistral(api_key=st.session_state.api_key)

# Function to load knowledge pyramid data
def load_pyramid_data():
    pyramid_data = {
        "recollections": [],
        "abstracts": [],
        "concepts": [],
        "insights": []
    }
    
    # Load recollections
    recollections_path = Path("knowledge_pyramid/recollections/recollections.txt")
    if recollections_path.exists():
        with open(recollections_path, 'r', encoding='utf-8') as f:
            pyramid_data["recollections"] = f.read()
    
    # Load abstracts
    abstracts_dir = Path("knowledge_pyramid/abstracts")
    if abstracts_dir.exists():
        for abstract_file in abstracts_dir.glob("*.txt"):
            with open(abstract_file, 'r', encoding='utf-8') as f:
                content = f.read()
                pyramid_data["abstracts"].append({
                    "source": abstract_file.stem,
                    "content": content
                })
    
    # Load concepts
    concepts_dir = Path("knowledge_pyramid/concepts")
    if concepts_dir.exists():
        for concept_file in concepts_dir.glob("*.json"):
            with open(concept_file, 'r', encoding='utf-8') as f:
                concepts = json.load(f)
                pyramid_data["concepts"].append({
                    "source": concept_file.stem,
                    "concepts": concepts
                })
    
    # Load insights
    insights_dir = Path("knowledge_pyramid/insights")
    if insights_dir.exists():
        for insight_file in insights_dir.glob("*.json"):
            with open(insight_file, 'r', encoding='utf-8') as f:
                insights = json.load(f)
                pyramid_data["insights"].append({
                    "source": insight_file.stem,
                    "insights": insights[:50]  # Limit to first 50 insights to avoid UI clutter
                })
    
    return pyramid_data

# Sidebar
with st.sidebar:
    st.markdown("<div class='main-header'>Document Pyramid Explorer</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # API Key input (optional override)
    api_key = st.text_input("Mistral API Key (Optional)", value=st.session_state.api_key, type="password")
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
    
    # File upload section
    st.markdown("<div class='sub-header'>Upload Document</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF, Excel, PowerPoint, or Word document", 
                                    type=["pdf", "xlsx", "xls", "pptx", "ppt", "docx", "doc"])
    
    # Process the uploaded file
    if uploaded_file is not None:
        # Create docs directory if it doesn't exist
        os.makedirs("docs", exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join("docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.file_uploaded = True
        st.session_state.current_file = uploaded_file.name
        st.success(f"File uploaded: {uploaded_file.name}")
    
    # Process document button
    if st.session_state.file_uploaded and not st.session_state.file_processed:
        if st.button("Process Document", key="process_doc"):
            with st.spinner("Processing document and extracting content..."):
                # Call the file_processor to process the document
                try:
                    # Clear existing outputs for clean processing
                    if os.path.exists("markdown_output"):
                        for file in os.listdir("markdown_output"):
                            if file.endswith(".md"):
                                os.remove(os.path.join("markdown_output", file))
                    
                    # Make sure the markdown_output directory exists
                    os.makedirs("markdown_output", exist_ok=True)
                    os.makedirs("markdown_output/images", exist_ok=True)
                    
                    # Set up the Mistral client for the file processor
                    client = get_mistral_client()
                    
                    # Process each document in the docs directory
                    doc_path = Path("docs") / st.session_state.current_file
                    document_text, images = file_processor.process_document(doc_path, client)
                    
                    # Create markdown with images
                    markdown_content = file_processor.create_markdown_with_images(
                        document_text, images, client, Path("markdown_output/images")
                    )
                    
                    # Write to output file
                    output_file = Path("markdown_output") / "processed_documents.md"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(markdown_content)
                    
                    st.session_state.file_processed = True
                    st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    
    # Build knowledge pyramid button
    if st.session_state.file_processed and not st.session_state.pyramid_built:
        if st.button("Build Knowledge Pyramid", key="build_pyramid"):
            with st.spinner("Building knowledge pyramid..."):
                try:
                    # Run the knowledge distillation process
                    knowledge_distillation.process_documents()
                    st.session_state.pyramid_built = True
                    st.success("Knowledge pyramid built successfully!")
                except Exception as e:
                    st.error(f"Error building knowledge pyramid: {str(e)}")
    
    # Information section
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. **Upload** your document
    2. **Process** to extract content
    3. **Build** the knowledge pyramid
    4. **Explore** the knowledge levels
    5. **Chat** with your document
    """)

# Main content area
st.markdown("<div class='main-header'>Document Understanding & Exploration</div>", unsafe_allow_html=True)

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Document Status", "Knowledge Pyramid", "Chat with Document"])

# Tab 1: Document Status
with tab1:
    st.markdown("<div class='sub-header'>Document Processing Status</div>", unsafe_allow_html=True)
    
    # Status cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Document Upload")
        if st.session_state.file_uploaded:
            st.markdown(f"✅ Uploaded: **{st.session_state.current_file}**")
        else:
            st.markdown("❌ No document uploaded")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Content Extraction")
        if st.session_state.file_processed:
            st.markdown("✅ Document processed")
            
            # Show preview button
            if st.button("Preview Processed Content"):
                markdown_path = Path("markdown_output/processed_documents.md")
                if markdown_path.exists():
                    try:
                        with open(markdown_path, 'r', encoding='utf-8') as f:
                            markdown_preview = f.read()[:1000] + "...\n\n[Content truncated]"
                        st.markdown(markdown_preview)
                    except UnicodeDecodeError:
                        # Try with different encodings if utf-8 fails
                        try:
                            with open(markdown_path, 'r', encoding='latin-1') as f:
                                markdown_preview = f.read()[:1000] + "...\n\n[Content truncated]"
                            st.markdown(markdown_preview)
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
                            st.info("The file contains characters that cannot be decoded properly. You can still proceed with the knowledge pyramid.")
        else:
            st.markdown("❌ Document not processed")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Knowledge Pyramid")
        if st.session_state.pyramid_built:
            st.markdown("✅ Pyramid built")
        else:
            st.markdown("❌ Pyramid not built")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Process steps visualization
    if st.session_state.file_uploaded:
        st.markdown("<div class='sub-header'>Processing Pipeline</div>", unsafe_allow_html=True)
        
        # Progress bar based on completed steps
        progress_value = 0
        if st.session_state.file_uploaded:
            progress_value += 0.33
        if st.session_state.file_processed:
            progress_value += 0.33
        if st.session_state.pyramid_built:
            progress_value += 0.34
        
        st.progress(progress_value)
        
        # Steps display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='section' style='background-color: #e3f2fd;'>", unsafe_allow_html=True)
            st.markdown("#### 1. Document Upload")
            st.markdown("✅ Complete")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            status = "✅ Complete" if st.session_state.file_processed else "⏳ Pending"
            color = "#e3f2fd" if st.session_state.file_processed else "#f5f5f5"
            st.markdown(f"<div class='section' style='background-color: {color};'>", unsafe_allow_html=True)
            st.markdown("#### 2. Content Extraction")
            st.markdown(status)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            status = "✅ Complete" if st.session_state.pyramid_built else "⏳ Pending"
            color = "#e3f2fd" if st.session_state.pyramid_built else "#f5f5f5"
            st.markdown(f"<div class='section' style='background-color: {color};'>", unsafe_allow_html=True)
            st.markdown("#### 3. Knowledge Pyramid")
            st.markdown(status)
            st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Knowledge Pyramid
with tab2:
    st.markdown("<div class='sub-header'>Knowledge Pyramid Explorer</div>", unsafe_allow_html=True)
    
    if st.session_state.pyramid_built:
        # Load pyramid data
        pyramid_data = load_pyramid_data()
        
        # Expandable sections for each pyramid level
        with st.expander("🔍 Recollections (High-Level Cross-Document Insights)", expanded=True):
            st.markdown("<div class='pyramid-level pyramid-recollections'>", unsafe_allow_html=True)
            st.markdown(pyramid_data["recollections"])
            st.markdown("</div>", unsafe_allow_html=True)
        
        with st.expander("📝 Abstracts (Document Summaries)"):
            for abstract in pyramid_data["abstracts"]:
                st.markdown(f"<div class='pyramid-level pyramid-abstracts'>", unsafe_allow_html=True)
                st.markdown(f"**Source: {abstract['source']}**")
                st.markdown(abstract["content"])
                st.markdown("</div>", unsafe_allow_html=True)
        
        with st.expander("🧩 Concepts (Higher-Level Themes)"):
            for concept_group in pyramid_data["concepts"]:
                st.markdown(f"<div class='pyramid-level pyramid-concepts'>", unsafe_allow_html=True)
                st.markdown(f"**Source: {concept_group['source']}**")
                for i, concept in enumerate(concept_group["concepts"]):
                    st.markdown(f"{i+1}. {concept}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with st.expander("💡 Insights (Atomic Facts)"):
            for insight_group in pyramid_data["insights"]:
                st.markdown(f"<div class='pyramid-level pyramid-insights'>", unsafe_allow_html=True)
                st.markdown(f"**Source: {insight_group['source']}**")
                for i, insight in enumerate(insight_group["insights"]):
                    st.markdown(f"{i+1}. {insight}")
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Knowledge pyramid has not been built yet. Please upload a document, process it, and build the pyramid.")

# Tab 3: Chat with Document
with tab3:
    st.markdown("<div class='sub-header'>Chat with Your Document</div>", unsafe_allow_html=True)
    
    if st.session_state.pyramid_built:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"<div class='chat-message user-message'>", unsafe_allow_html=True)
                st.markdown(f"**You:** {message['content']}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message assistant-message'>", unsafe_allow_html=True)
                st.markdown(f"**Assistant:** {message['content']}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Chat input
        user_query = st.text_input("Ask a question about your document", key="user_query")
        
        if user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Process query
            with st.spinner("Searching knowledge pyramid for answers..."):
                try:
                    response = query_pyramid.process_query(user_query)
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Force a rerun to update the chat display
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
    else:
        st.info("Please build the knowledge pyramid first before chatting with your document.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Saif Bagmaru")