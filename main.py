# %% [markdown]
# # Geshe Kelsang Gyatso Teachings Explorer
# 
# This notebook creates a RAG (Retrieval Augmented Generation) system for exploring the teachings of 
# Geshe Kelsang Gyatso. It processes EPUB files of his works, creates embeddings, and allows you to 
# have conversations that draw directly from his teachings with proper citations.
# 
# ## Features:
# - Conversation memory that builds on past interactions
# - Enhanced citations with proper formatting
# - Quality feedback tracking
# - Session management
# - Export functionality
# - Smart chunking for better context

# %% [markdown]
# ## Step 1: Setup Environment
# Run this cell to install packages and set up the environment

# %%
# SETUP CELL - RUN THIS FIRST
# ==========================
# Just click the play button (▶️) on the left to run

print("Setting up the Geshe Kelsang Gyatso Teachings Explorer...")

# Install required packages
!pip install openai chromadb ebooklib beautifulsoup4 tiktoken anthropic tqdm python-dotenv ipywidgets markdown numpy scipy -q

# Import necessary libraries
import os
import json
import glob
import re
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import openai
import tiktoken
import time
import logging
import numpy as np
from tqdm.notebook import tqdm
from google.colab import drive, output
from dotenv import load_dotenv
import ipywidgets as ipyw
from IPython.display import display, HTML, clear_output, FileLink
import markdown
import chromadb
import anthropic
from anthropic import Anthropic
from datetime import datetime
import threading
import uuid
import math
from scipy.spatial.distance import cosine

# Mount Google Drive to access files
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Define directory structure
BASE_DIR = "/content/drive/MyDrive/master_rag"
EPUB_DIR = f"{BASE_DIR}/epub_directory"  # Using the specified subdirectory
TEXT_DIR = f"{BASE_DIR}/extracted_text"
EMBEDDINGS_DIR = f"{BASE_DIR}/embeddings"
VECTORDB_DIR = f"{BASE_DIR}/vector_db"
LOG_DIR = f"{BASE_DIR}/logs"
EXPORT_DIR = f"{BASE_DIR}/exports"
SESSION_DIR = f"{BASE_DIR}/sessions"
HISTORY_DIR = f"{BASE_DIR}/history"

# Create directories
for dir_path in [BASE_DIR, TEXT_DIR, EMBEDDINGS_DIR, VECTORDB_DIR, LOG_DIR, EXPORT_DIR, SESSION_DIR, HISTORY_DIR]:
    os.makedirs(dir_path, exist_ok=True)

print(f"✅ Using EPUB directory: {EPUB_DIR}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.FileHandler(f"{LOG_DIR}/geshe_teachings.log")  # File handler
    ]
)
logger = logging.getLogger(__name__)

# Create .env file template if it doesn't exist
env_file_path = f"{BASE_DIR}/.env"
if not os.path.exists(env_file_path):
    with open(env_file_path, 'w') as f:
        f.write("""# API Keys for Geshe Kelsang Gyatso Teachings Explorer
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
""")
    print(f"✅ Created .env template at {env_file_path}")
    print("Please edit this file to add your actual API keys before proceeding.")
else:
    print(f"✅ Found existing .env file at {env_file_path}")

# Load API keys from .env file
load_dotenv(env_file_path)

# Setup OpenAI client
openai_client = None
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    print("\n⚠️ OpenAI API key not found. Please edit the .env file in this folder:")
    print(f"{BASE_DIR}")
    print("Add your OpenAI API key to the file, replacing the placeholder text.")
else:
    print("✅ OpenAI API key found")
    openai_client = openai.OpenAI(api_key=openai_api_key)

# Check for Anthropic API key
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
if not anthropic_api_key:
    print("\n⚠️ Anthropic API key not found. For best results, edit the .env file")
    print(f"in {BASE_DIR} and add your Anthropic API key, replacing the placeholder text.")
else:
    print("✅ Anthropic API key found")

# Check if there are EPUB files in the directory
epub_files = glob.glob(f"{EPUB_DIR}/*.epub")
if len(epub_files) == 0:
    print(f"\n⚠️ No EPUB files found in {EPUB_DIR}")
    print(f"Please add EPUB files of Geshe Kelsang Gyatso's teachings to this folder.")
else:
    print(f"✅ Found {len(epub_files)} EPUB files in {EPUB_DIR}")

# Initialize global variables
current_user_name = None

print("\n✅ Setup complete! Now run the next cell to start exploring the teachings.")

# %% [markdown]
# ## Text Processing Functions
# These functions extract and process text from the EPUB files

# %%
# TEXT PROCESSING FUNCTIONS
# ========================

def improved_chunking(text, max_tokens=4000, overlap=200):
    """Split text into chunks at natural boundaries where possible"""
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Split into paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = encoding.encode(para)
        para_token_count = len(para_tokens)
        
        # If adding this paragraph would exceed max_tokens
        if current_tokens + para_token_count > max_tokens and current_tokens > 0:
            # Complete this chunk
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(chunk_text)
            
            # Start a new chunk with overlap
            # Find paragraphs that fit within overlap token count
            overlap_tokens = 0
            overlap_paras = []
            for prev_para in reversed(current_chunk):
                prev_tokens = len(encoding.encode(prev_para))
                if overlap_tokens + prev_tokens <= overlap:
                    overlap_paras.insert(0, prev_para)
                    overlap_tokens += prev_tokens
                else:
                    break
            
            # Reset with overlap paragraphs
            current_chunk = overlap_paras.copy()
            current_tokens = overlap_tokens
        
        # Add the paragraph to the current chunk
        current_chunk.append(para)
        current_tokens += para_token_count
    
    # Add the last chunk if there's anything left
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunks.append(chunk_text)
    
    return chunks

def extract_text_with_metadata(epub_path):
    """Extract text from EPUB while preserving metadata about source and structure"""
    logger.info(f"Extracting text from {os.path.basename(epub_path)}")
    try:
        book = epub.read_epub(epub_path)
        book_title = "Unknown Title"
        try:
            title_data = book.get_metadata('DC', 'title')
            if title_data and len(title_data) > 0 and len(title_data[0]) > 0:
                book_title = title_data[0][0]
        except Exception as e:
            logger.warning(f"Could not extract title: {e}")
            
        book_id = os.path.basename(epub_path).replace('.epub', '')
        
        # Extract creator if available
        creator = "Geshe Kelsang Gyatso"
        try:
            creator_data = book.get_metadata('DC', 'creator')
            if creator_data and len(creator_data) > 0 and len(creator_data[0]) > 0:
                creator = creator_data[0][0]
        except Exception as e:
            logger.warning(f"Could not extract creator: {e}")
        
        chapters = []
        # Track current position for page number estimation
        current_position = 0
        position_to_page = {}  # Map character positions to estimated page numbers
        chars_per_page = 2000  # Approximate characters per page
        
        total_items = len(list(book.get_items()))
        processed_items = 0
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                processed_items += 1
                if processed_items % 10 == 0:
                    logger.info(f"Processing item {processed_items}/{total_items} in {book_title}")
                    
                try:
                    content = item.get_content().decode('utf-8', errors='replace')
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Try to extract chapter/section title
                    chapter_title = None
                    heading = soup.find(['h1', 'h2', 'h3'])
                    if heading:
                        chapter_title = heading.get_text().strip()
                    
                    # Extract text content
                    text = soup.get_text()
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    if text:
                        # Calculate page numbers (estimation)
                        for i in range(0, len(text), chars_per_page):
                            position_to_page[current_position + i] = (current_position + i) // chars_per_page + 1
                        
                        # Add chapter with metadata
                        chapters.append({
                            "content": text,
                            "chapter_title": chapter_title,
                            "start_position": current_position,
                        })
                        
                        current_position += len(text)
                except Exception as e:
                    logger.error(f"Error processing item in {book_title}: {e}")
        
        logger.info(f"Completed extraction for {book_title}: {len(chapters)} chapters, {current_position} characters")
        
        return {
            "book_id": book_id,
            "book_title": book_title,
            "creator": creator,
            "chapters": chapters,
            "position_to_page": position_to_page,
            "total_length": current_position
        }
    except Exception as e:
        logger.error(f"Error processing EPUB {epub_path}: {e}")
        return None

def split_into_chunks_with_metadata(book_data, max_tokens=4000, overlap=200):
    """Split book text into chunks while preserving metadata"""
    logger.info(f"Chunking text for {book_data['book_title']}")
    encoding = tiktoken.get_encoding("cl100k_base")
    
    chunks = []
    total_chapters = len(book_data["chapters"])
    
    for chapter_idx, chapter in enumerate(book_data["chapters"]):
        if chapter_idx % 5 == 0 or chapter_idx == total_chapters - 1:
            logger.info(f"Chunking chapter {chapter_idx + 1}/{total_chapters} in {book_data['book_title']}")
            
        text = chapter["content"]
        start_position = chapter["start_position"]
        chapter_title = chapter["chapter_title"]
        
        # Use improved chunking
        text_chunks = improved_chunking(text, max_tokens, overlap)
        
        # Track current position within the chapter
        current_pos = 0
        
        for i, chunk_text in enumerate(text_chunks):
            # Calculate chunk position in the book
            chunk_start_pos = start_position + current_pos
            chunk_end_pos = chunk_start_pos + len(chunk_text)
            current_pos += len(chunk_text) - (overlap if i < len(text_chunks) - 1 else 0)
            
            # Estimate page numbers
            start_page = 1
            end_page = 1
            for pos, page in book_data["position_to_page"].items():
                if pos <= chunk_start_pos:
                    start_page = page
                if pos <= chunk_end_pos:
                    end_page = page
                else:
                    break
            
            # Create chunk with metadata
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "book_id": book_data["book_id"],
                    "book_title": book_data["book_title"],
                    "chapter_title": chapter_title,
                    "start_page": start_page,
                    "end_page": end_page,
                    "chunk_index": len(chunks)
                }
            })
    
    logger.info(f"Created {len(chunks)} chunks for {book_data['book_title']}")
    return chunks

def create_embeddings_batch(chunks, batch_size=100):
    """Create embeddings for text chunks in batches with detailed progress tracking"""
    global openai_client
    
    if not openai_client:
        logger.error("OpenAI client not configured")
        return []
    
    all_embeddings = []
    total_chunks = len(chunks)
    
    print(f"Creating embeddings for {total_chunks} chunks:")
    progress_bar = tqdm(total=total_chunks, desc="Embedding progress", unit="chunk")
    
    # Process in batches
    batch_count = 0
    for i in range(0, total_chunks, batch_size):
        batch_count += 1
        end_idx = min(i + batch_size, total_chunks)
        batch = chunks[i:end_idx]
        batch_size_actual = len(batch)
        
        logger.info(f"Processing batch {batch_count}: chunks {i+1}-{end_idx} of {total_chunks}")
        print(f"Batch {batch_count}: Processing chunks {i+1}-{end_idx} of {total_chunks}")
        
        retry_count = 0
        max_retries = 5
        success = False
        
        while not success and retry_count < max_retries:
            try:
                # Extract just the text for embedding
                texts = [chunk["text"] for chunk in batch]
                
                # Create embeddings
                response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts
                )
                
                # Add embeddings to chunks
                for j, embedding_data in enumerate(response.data):
                    chunk_with_embedding = batch[j].copy()
                    chunk_with_embedding["embedding"] = embedding_data.embedding
                    all_embeddings.append(chunk_with_embedding)
                
                # Update progress
                progress_bar.update(batch_size_actual)
                
                # Batch successful
                success = True
                logger.info(f"Successfully embedded batch {batch_count} ({batch_size_actual} chunks)")
                
                # Add delay to respect rate limits
                if end_idx < total_chunks:
                    time.sleep(0.5)
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Error creating embeddings for batch {batch_count} (attempt {retry_count}/{max_retries}): {e}")
                print(f"⚠️ Error in batch {batch_count}: {str(e)[:100]}... Retrying ({retry_count}/{max_retries})")
                
                # Wait longer if we hit rate limits
                if "rate limit" in str(e).lower():
                    wait_time = 60 * retry_count  # Increase wait time with each retry
                    logger.info(f"Rate limit hit, waiting {wait_time} seconds...")
                    print(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # Other errors
                    time.sleep(5)
        
        if not success:
            logger.error(f"Failed to process batch {batch_count} after {max_retries} attempts")
            print(f"❌ Failed to process batch {batch_count} after {max_retries} attempts. Continuing with next batch.")
    
    progress_bar.close()
    logger.info(f"Embedding complete: {len(all_embeddings)}/{total_chunks} chunks embedded successfully")
    print(f"Embedding complete: {len(all_embeddings)}/{total_chunks} chunks embedded successfully")
    
    return all_embeddings

def create_vector_database(chunks_with_embeddings, collection_name="geshe_kelsang_gyatso"):
    """Create a Chroma vector database from chunks with embeddings"""
    logger.info("Creating vector database")
    print("Creating vector database...")
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=VECTORDB_DIR)
    
    # Create or get collection
    try:
        # Try to get existing collection
        collection = chroma_client.get_collection(name=collection_name)
        logger.info(f"Found existing collection: {collection_name}")
        
        # Delete and recreate collection to ensure clean state
        logger.info(f"Deleting existing collection to recreate with new data")
        chroma_client.delete_collection(name=collection_name)
        collection = chroma_client.create_collection(name=collection_name)
        logger.info(f"Recreated collection: {collection_name}")
    except:
        # Create new collection if it doesn't exist
        collection = chroma_client.create_collection(name=collection_name)
        logger.info(f"Created new collection: {collection_name}")
    
    # Prepare data for insertion
    total_chunks = len(chunks_with_embeddings)
    logger.info(f"Preparing {total_chunks} chunks for database insertion")
    
    ids = [f"chunk_{chunk['metadata']['book_id']}_{chunk['metadata']['chunk_index']}" for chunk in chunks_with_embeddings]
    documents = [chunk["text"] for chunk in chunks_with_embeddings]
    embeddings = [chunk["embedding"] for chunk in chunks_with_embeddings]
    metadatas = [chunk["metadata"] for chunk in chunks_with_embeddings]
    
    # Add to collection in batches
    batch_size = 100
    total_batches = (total_chunks + batch_size - 1) // batch_size
    
    progress_bar = tqdm(total=total_batches, desc="Database insertion", unit="batch")
    
    for i in range(0, total_chunks, batch_size):
        batch_num = i // batch_size + 1
        end_idx = min(i + batch_size, total_chunks)
        
        logger.info(f"Adding batch {batch_num}/{total_batches} to vector database ({i+1}-{end_idx} of {total_chunks} chunks)")
        
        try:
            collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            progress_bar.update(1)
            
        except Exception as e:
            logger.error(f"Error adding batch {batch_num} to database: {e}")
            print(f"❌ Error adding batch {batch_num} to database: {str(e)[:100]}...")
    
    progress_bar.close()
    logger.info(f"Vector database creation complete. Collection: {collection_name}")
    print(f"✅ Vector database creation complete!")
    
    return collection

# %% [markdown]
# ## Main Processing Function
# This processes all EPUB files and creates the vector database

# %%
# MAIN PROCESSING FUNCTION
# =======================

def process_all_epubs():
    """Process all EPUB files in the directory"""
    logger.info("Starting EPUB processing")
    
    # Get list of EPUB files
    epub_files = glob.glob(f"{EPUB_DIR}/*.epub")
    
    if len(epub_files) == 0:
        msg = f"No EPUB files found in {EPUB_DIR}. Please add some EPUB files before processing."
        logger.error(msg)
        return msg
    
    print(f"Found {len(epub_files)} EPUB files to process:")
    for epub_file in epub_files:
        print(f"  - {os.path.basename(epub_file)}")
    
    all_chunks = []
    
    # Process each EPUB file with progress bar
    for i, epub_file in enumerate(tqdm(epub_files, desc="Processing books", unit="book")):
        book_name = os.path.basename(epub_file).replace('.epub', '')
        logger.info(f"Processing book {i+1}/{len(epub_files)}: {book_name}")
        print(f"\nProcessing book {i+1}/{len(epub_files)}: {book_name}")
        
        # Extract text with metadata
        book_data = extract_text_with_metadata(epub_file)
        
        if not book_data:
            logger.error(f"Failed to extract text from {book_name}. Skipping.")
            print(f"❌ Failed to extract text from {book_name}. Skipping.")
            continue
        
        # Save extracted text
        text_path = f"{TEXT_DIR}/{book_name}.json"
        try:
            with open(text_path, 'w') as f:
                json.dump(book_data, f)
            logger.info(f"Saved extracted text to {text_path}")
        except Exception as e:
            logger.error(f"Error saving extracted text: {e}")
            print(f"⚠️ Error saving extracted text: {str(e)[:100]}...")
        
        # Split into chunks
        try:
            chunks = split_into_chunks_with_metadata(book_data)
            all_chunks.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks from {book_name}")
            print(f"✅ Created {len(chunks)} chunks from {book_name}")
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            print(f"❌ Error creating chunks: {str(e)[:100]}...")
    
    # Create embeddings
    print(f"\n{'='*50}")
    logger.info(f"Starting embedding generation for {len(all_chunks)} chunks")
    print(f"Starting embedding generation for {len(all_chunks)} chunks")
    print(f"{'='*50}\n")
    
    chunks_with_embeddings = create_embeddings_batch(all_chunks)
    
    if not chunks_with_embeddings:
        logger.error("No embeddings were created. Check API key and connection.")
        return "Failed to create embeddings. Check your OpenAI API key and internet connection."
    
    # Save embeddings
    logger.info(f"Saving {len(chunks_with_embeddings)} embeddings")
    print(f"Saving {len(chunks_with_embeddings)} embeddings...")
    
    try:
        embeddings_path = f"{EMBEDDINGS_DIR}/all_embeddings.json"
        with open(embeddings_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_chunks = []
            for chunk in chunks_with_embeddings:
                chunk_copy = chunk.copy()
                chunk_copy["embedding"] = chunk["embedding"] if isinstance(chunk["embedding"], list) else chunk["embedding"].tolist()
                serializable_chunks.append(chunk_copy)
            json.dump(serializable_chunks, f)
        logger.info(f"Saved embeddings to {embeddings_path}")
        print(f"✅ Saved embeddings to {embeddings_path}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")
        print(f"⚠️ Error saving embeddings: {str(e)[:100]}...")
    
    # Create vector database
    print(f"\n{'='*50}")
    print("Creating vector database...")
    print(f"{'='*50}\n")
    
    try:
        collection = create_vector_database(chunks_with_embeddings)
        logger.info("Processing complete!")
        print("\n✅ Processing complete!")
        return collection
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        print(f"❌ Error creating vector database: {str(e)[:100]}...")
        return f"Error creating vector database: {str(e)}"

# %% [markdown]
# ## History and Context Management
# These functions handle saving and using conversation history

# %%
# HISTORY AND CONTEXT MANAGEMENT
# =============================

def save_interaction(user_name, query, response, search_results=None):
    """Save an interaction to the history file"""
    # Create a unique ID for this interaction
    interaction_id = str(uuid.uuid4())
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract sources from search results
    sources = []
    if search_results and "metadatas" in search_results and search_results["metadatas"]:
        for metadata in search_results["metadatas"][0]:
            sources.append({
                "book_title": metadata["book_title"],
                "chapter": metadata.get("chapter_title", ""),
                "start_page": metadata["start_page"],
                "end_page": metadata["end_page"]
            })
    
    # Create interaction data
    interaction = {
        "id": interaction_id,
        "timestamp": timestamp,
        "user_name": user_name,
        "query": query,
        "response": response,
        "sources": sources
    }
    
    # Create embedding for query to enable similarity search later
    try:
        if openai_client:
            query_response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            interaction["query_embedding"] = query_response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding for history: {e}")
        # Continue without embedding
    
    # File to store user-specific history
    history_file = f"{HISTORY_DIR}/{user_name.lower().replace(' ', '_')}_history.json"
    
    # Load existing history or create new
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try:
                history = json.load(f)
            except:
                history = {"interactions": []}
    else:
        history = {"interactions": []}
    
    # Add new interaction
    history["interactions"].append(interaction)
    
    # Save updated history
    with open(history_file, 'w') as f:
        # Convert embeddings to lists for JSON serialization
        for inter in history["interactions"]:
            if "query_embedding" in inter and not isinstance(inter["query_embedding"], list):
                inter["query_embedding"] = inter["query_embedding"].tolist()
        json.dump(history, f, indent=2)
    
    logger.info(f"Saved interaction for {user_name}: {query[:50]}...")
    return interaction_id

def find_related_past_interactions(user_name, current_query, limit=3):
    """Find past interactions related to the current query"""
    history_file = f"{HISTORY_DIR}/{user_name.lower().replace(' ', '_')}_history.json"
    
    if not os.path.exists(history_file):
        return []
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # If no history, return empty list
        if not history["interactions"]:
            return []
        
        # Get embedding for current query
        query_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=current_query
        )
        query_embedding = query_response.data[0].embedding
        
        # Calculate similarity for each past interaction
        similarities = []
        for i, interaction in enumerate(history["interactions"]):
            if "query_embedding" not in interaction:
                continue
                
            past_embedding = interaction["query_embedding"]
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(query_embedding, past_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top related interactions
        related = []
        for idx, score in similarities[:limit]:
            if score > 0.75:  # Only include if similarity is high enough
                interaction = history["interactions"][idx]
                related.append({
                    "query": interaction["query"],
                    "response": interaction["response"],
                    "timestamp": interaction["timestamp"],
                    "similarity": score
                })
        
        return related
    
    except Exception as e:
        logger.error(f"Error finding related interactions: {e}")
        return []

def get_conversation_context(user_name, current_query):
    """Get context from previous conversations to include in prompt"""
    related = find_related_past_interactions(user_name, current_query)
    
    if not related:
        return ""
    
    # Format the context
    context = "I've found some related questions you've asked before:\n\n"
    
    for i, item in enumerate(related):
        # Calculate time difference
        past_time = datetime.strptime(item["timestamp"], "%Y-%m-%d %H:%M:%S")
        current_time = datetime.now()
        time_diff = current_time - past_time
        
        days_ago = time_diff.days
        if days_ago > 365:
            time_text = f"about {days_ago // 365} year(s) ago"
        elif days_ago > 30:
            time_text = f"about {days_ago // 30} month(s) ago"
        elif days_ago > 0:
            time_text = f"{days_ago} day(s) ago"
        else:
            time_text = "earlier today"
        
        context += f"{i+1}. {time_text}, you asked: \"{item['query']}\"\n"
        # Include a snippet of the previous response
        response_snippet = item["response"]
        if len(response_snippet) > 300:
            response_snippet = response_snippet[:300] + "..."
        context += f"   My response included: \"{response_snippet}\"\n\n"
    
    return context

def reset_for_new_user():
    """Reset the system for a new user while preserving the processed database"""
    global current_user_name
    
    logger.info("Resetting for new user")
    
    # Clear current user
    current_user_name = None
    
    # Restart the explorer
    clear_output()
    run_teaching_explorer()
    
    return "System reset for new user"

def format_citation(metadata):
    """Format citation in a consistent way"""
    citation = f"[{metadata['book_title']}"
    if metadata['start_page'] == metadata['end_page']:
        citation += f", p.{metadata['start_page']}"
    else:
        citation += f", pp.{metadata['start_page']}-{metadata['end_page']}"
    citation += "]"
    return citation

def extract_topics(response, query):
    """Extract key Buddhist topics from response"""
    common_topics = [
        "compassion", "emptiness", "meditation", "bodhichitta", 
        "karma", "dharma", "lamrim", "tantra", "enlightenment",
        "wisdom", "mindfulness", "concentration", "ethics",
        "renunciation", "liberation", "samsara", "nirvana",
        "buddha", "attachment", "impermanence", "suffering"
    ]
    
    found_topics = []
    combined_text = (response.lower() + ' ' + query.lower())
    
    for topic in common_topics:
        if re.search(r'\b' + topic + r'\b', combined_text):
            found_topics.append(topic)
    
    return found_topics

def display_topics(topics):
    """Display topic tags in a visually appealing way"""
    if not topics:
        return ""
        
    html = "<div style='margin-top:10px;'><span style='font-weight:bold;'>Topics: </span>"
    for topic in topics:
        html += f"<span style='background:#e3f2fd;padding:3px 8px;border-radius:12px;margin-right:5px;font-size:0.9em;'>{topic}</span>"
    html += "</div>"
    return html

def highlight_quotes_in_response(response):
    """Find quoted text and highlight it visually"""
    # Look for text in quotation marks
    quote_pattern = r'"([^"]+)"'
    highlighted = re.sub(
        quote_pattern,
        r'<span style="background-color:#e8f5e9;font-style:italic;">&ldquo;\1&rdquo;</span>',
        response
    )
    return highlighted

def display_similarity_info(results):
    """Show a simple visual indicator of passage relevance"""
    html = "<div style='margin-top:10px;font-size:0.9em;color:#666;'><i>Sources by relevance:</i><br>"
    
    for i, (metadata, distance) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
        # Convert distance to similarity (0-100%)
        similarity = int((1 - distance) * 100)
        
        # Create a visual bar
        bar_width = similarity
        bar_color = "#4CAF50" if similarity > 80 else "#FFC107" if similarity > 60 else "#F44336"
        
        html += f"{metadata['book_title']} (p.{metadata['start_page']}): "
        html += f"<span style='display:inline-block;width:{bar_width}px;height:8px;background:{bar_color};'></span> {similarity}%<br>"
    
    html += "</div>"
    return html

def add_export_option(response, query):
    """Add button to export the response"""
    # Create unique filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    sanitized_query = re.sub(r'[^\w\s]', '', query)[:30].strip().replace(' ', '_')
    filename = f"response_{sanitized_query}_{timestamp}"
    
    # Create HTML version
    html_filename = f"{filename}.html"
    html_path = f"{EXPORT_DIR}/{html_filename}"
    
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    with open(html_path, 'w') as f:
        f.write(f"""
        <html>
        <head>
            <title>Response: {query}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .query {{ font-weight: bold; background: #f5f5f5; padding: 10px; border-left: 3px solid #03A9F4; }}
                .response {{ margin-top: 20px; }}
                .citation {{ font-style: italic; color: #666; }}
            </style>
        </head>
        <body>
            <h1>Geshe Kelsang Gyatso Teachings</h1>
            <div class="query">{query}</div>
            <div class="response">{markdown.markdown(response)}</div>
            <hr>
            <p>Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        </body>
        </html>
        """)
    
    # Create a FileLink to the exported file
    file_link = FileLink(html_path)
    
    export_html = f"""
    <div style="margin-top:10px;">
        <p>Download response as HTML:</p>
        {file_link._repr_html_()}
    </div>
    """
    return export_html

def add_feedback_buttons(response_id, query, response):
    """Add feedback buttons that save the feedback data"""
    # Create a unique ID for this feedback
    timestamp = int(time.time())
    feedback_id = f"{timestamp}_{response_id}"
    
    # Create the feedback HTML with proper saving
    feedback_html = f"""
    <div style="margin-top:10px;">
        <p>Was this response helpful?</p>
        <button onclick="save_feedback_{feedback_id}('helpful')" style="background:#4CAF50;color:white;border:none;padding:5px 10px;border-radius:4px;margin-right:10px;">Yes</button>
        <button onclick="save_feedback_{feedback_id}('not_helpful')" style="background:#f44336;color:white;border:none;padding:5px 10px;border-radius:4px;">No</button>
    </div>
    
    <script>
    function save_feedback_{feedback_id}(value) {{
        // Create feedback data
        const feedback_data = {{
            'id': '{feedback_id}',
            'query': '{query.replace("'", "\\'")}',
            'value': value,
            'timestamp': Date.now()
        }};
        
        // Use Google Colab's communication API to pass data to Python
        google.colab.kernel.invokeFunction(
            'saveFeedback', // Function name
            [JSON.stringify(feedback_data)], // Arguments
            {{}} // Callbacks
        );
        
        // Thank the user
        document.getElementById('feedback_msg_{feedback_id}').innerHTML = 'Thank you for your feedback!';
    }}
    </script>
    <div id="feedback_msg_{feedback_id}" style="margin-top:5px;font-style:italic;"></div>
    """
    
    # Register the Python callback function
    def save_feedback_to_file(feedback_json):
        """Save feedback data to a file in Google Drive"""
        feedback_data = json.loads(feedback_json)
        
        # File to store feedback
        feedback_file = f"{BASE_DIR}/user_feedback.json"
        
        # Load existing feedback or create new structure
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                try:
                    all_feedback = json.load(f)
                except:
                    all_feedback = {"feedback": []}
        else:
            all_feedback = {"feedback": []}
        
        # Add more data to the feedback
        feedback_data["response_snippet"] = response[:200] + "..." if len(response) > 200 else response
        
        # Append new feedback
        all_feedback["feedback"].append(feedback_data)
        
        # Save updated feedback
        with open(feedback_file, 'w') as f:
            json.dump(all_feedback, f, indent=2)
            
        # Log feedback saved
        logger.info(f"Saved feedback: {feedback_data['value']} for query: {feedback_data['query'][:50]}...")
    
    # Register the callback
    output.register_callback('saveFeedback', save_feedback_to_file)
    
    return feedback_html

def analyze_feedback():
    """Analyze collected feedback to improve the system"""
    feedback_file = f"{BASE_DIR}/user_feedback.json"
    
    if not os.path.exists(feedback_file):
        return "No feedback data available yet."
    
    with open(feedback_file, 'r') as f:
        all_feedback = json.load(f)
    
    if not all_feedback["feedback"]:
        return "No feedback entries found."
    
    # Count positive and negative feedback
    total = len(all_feedback["feedback"])
    helpful = sum(1 for item in all_feedback["feedback"] if item["value"] == "helpful")
    not_helpful = total - helpful
    
    # Calculate percentage
    helpful_pct = (helpful / total) * 100 if total > 0 else 0
    
    # Find patterns in negative feedback
    negative_feedback = [item for item in all_feedback["feedback"] if item["value"] == "not_helpful"]
    
    # Analyze common words/phrases in negative feedback queries
    if negative_feedback:
        negative_queries = [item["query"] for item in negative_feedback]
        # Simple analysis - count word frequency
        word_counts = {}
        for query in negative_queries:
            words = query.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Find most common words
        common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    else:
        common_words = []
    
    # Generate report
    report = f"""
    # Feedback Analysis
    
    ## Overall Statistics
    - Total feedback entries: {total}
    - Helpful responses: {helpful} ({helpful_pct:.1f}%)
    - Not helpful responses: {not_helpful} ({100-helpful_pct:.1f}%)
    
    ## Recent Feedback
    """
    
    # Add recent feedback entries
    for item in all_feedback["feedback"][-5:]:
        report += f"""
    - **{item['value']}**: "{item['query'][:50] + '...' if len(item['query']) > 50 else item['query']}"
      - Response: "{item['response_snippet'][:100] + '...' if len(item['response_snippet']) > 100 else item['response_snippet']}"
    """
    
    # Add common issue words if available
    if common_words:
        report += "\n## Common Words in Negative Feedback\n"
        for word, count in common_words:
            report += f"- {word}: {count} occurrences\n"
    
    return report

def display_feedback_analysis():
    """Display analysis of collected feedback"""
    analysis = analyze_feedback()
    
    # Convert markdown to HTML
    html_analysis = markdown.markdown(analysis)
    
    # Display in a styled div
    display(HTML(f"""
    <div style="padding:15px; background-color:#f5f5f5; border-radius:5px; margin:20px 0;">
        {html_analysis}
    </div>
    """))

def save_session(name, user_name):
    """Save current session"""
    if not name or not user_name:
        return "Invalid session name or user name"
    
    # Get user history
    history_file = f"{HISTORY_DIR}/{user_name.lower().replace(' ', '_')}_history.json"
    if not os.path.exists(history_file):
        return "No history found to save"
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # Save as session
        session_file = f"{SESSION_DIR}/{name.replace(' ', '_')}.json"
        with open(session_file, 'w') as f:
            json.dump({
                "user_name": user_name,
                "session_name": name,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "history": history
            }, f, indent=2)
        
        logger.info(f"Saved session '{name}' for user {user_name}")
        return f"Session '{name}' saved successfully"
    except Exception as e:
        logger.error(f"Error saving session: {e}")
        return f"Error saving session: {str(e)}"

def load_session(session_name):
    """Load a saved session"""
    if not session_name:
        return "Invalid session name"
    
    session_file = f"{SESSION_DIR}/{session_name.replace(' ', '_')}.json"
    if not os.path.exists(session_file):
        return f"Session '{session_name}' not found"
    
    try:
        with open(session_file, 'r') as f:
            session = json.load(f)
        
        user_name = session["user_name"]
        history = session["history"]
        
        # Save history to user's history file
        history_file = f"{HISTORY_DIR}/{user_name.lower().replace(' ', '_')}_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Loaded session '{session_name}' for user {user_name}")
        
        # Set current user
        global current_user_name
        current_user_name = user_name
        
        return f"Session '{session_name}' loaded successfully for {user_name}"
    except Exception as e:
        logger.error(f"Error loading session: {e}")
        return f"Error loading session: {str(e)}"

# %% [markdown]
# ## Query and Response Functions
# These functions handle searching the vector database and generating responses

# %%
# QUERY AND RESPONSE FUNCTIONS
# ===========================

def setup_anthropic_client():
    """Set up Anthropic client for Claude"""
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key:
        logger.warning("Anthropic API key not found")
        return None
    
    try:
        client = Anthropic(api_key=anthropic_api_key)
        logger.info("Anthropic client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Error setting up Anthropic client: {e}")
        return None

def query_master_teachings(user_query, vector_collection, top_k=5):
    """Query the vector database for relevant teachings"""
    global openai_client
    
    logger.info(f"Querying for: '{user_query}' (top {top_k} results)")
    
    try:
        # Create embedding for the query
        query_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=user_query
        )
        query_embedding = query_response.data[0].embedding
        
        # Search the vector database
        results = vector_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        logger.info(f"Found {len(results['documents'][0])} relevant passages")
        return results
    
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise e

def generate_response_with_claude(user_query, search_results, anthropic_client, include_detailed_citations=True, user_name="Friend", use_conversation_history=True):
    """Generate a response using Claude, prioritizing the master's teachings"""
    if not anthropic_client:
        logger.warning("Anthropic client not available, using fallback response generator")
        # Fallback to generating a response without Claude
        return generate_fallback_response(user_query, search_results, user_name)
    
    logger.info(f"Generating response with Claude for '{user_query}'")
    
    # Format the context from search results
    context = "The following are excerpts from Geshe Kelsang Gyatso's teachings:\n\n"
    
    for i, (doc, metadata) in enumerate(zip(search_results["documents"][0], search_results["metadatas"][0])):
        context += f"Excerpt {i+1} from '{metadata['book_title']}'"
        
        if metadata.get('chapter_title'):
            context += f", chapter: {metadata['chapter_title']}"
            
        context += f", pages {metadata['start_page']}-{metadata['end_page']}:\n{doc}\n\n"
    
    # Get conversation history context if enabled
    history_context = ""
    if use_conversation_history:
        history_context = get_conversation_context(user_name, user_query)
    
    # Create prompt for Claude
    citation_instruction = """
    When providing citations, use footnote style with book title and page numbers, 
    for example: [Modern Buddhism, p.45-46]
    
    When quoting directly from Geshe Kelsang Gyatso's teachings, always use quotation marks around the direct quotes.
    """ if include_detailed_citations else """
    Include minimal citations, just mentioning the book title when referencing specific teachings.
    """
    
    prompt = f"""
    You are a system that helps users explore and understand the teachings of Geshe Kelsang Gyatso, a Tibetan Buddhist master.
    When answering questions, always prioritize using the provided excerpts from his works.
    
    The user's name is {user_name}. Address them by name in your response in a warm, friendly manner.
    
    {context}
    
    {history_context}
    
    User question: {user_query}
    
    Please answer the question based primarily on Geshe Kelsang Gyatso's teachings provided above.
    Only use your general knowledge if the passages don't contain relevant information.
    
    {citation_instruction}
    
    If asked about where a specific topic is discussed in his works, focus on providing those references specifically.
    """
    
    # Get response from Claude
    try:
        response = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        logger.info(f"Successfully generated response with Claude")
        return response.content
    except Exception as e:
        logger.error(f"Error generating response with Claude: {e}")
        return generate_fallback_response(user_query, search_results, user_name)

def generate_fallback_response(user_query, search_results, user_name="Friend"):
    """Generate a response without Claude when the API is unavailable"""
    logger.info(f"Generating fallback response for '{user_query}'")
    
    # Extract relevant passages
    passages = []
    sources = []
    
    for i, (doc, metadata) in enumerate(zip(search_results["documents"][0], search_results["metadatas"][0])):
        passages.append(doc)
        sources.append(f"{metadata['book_title']}, pages {metadata['start_page']}-{metadata['end_page']}")
    
    # Create a simple response
    response = f"Hello {user_name},\n\n"
    response += f"Here's what I found in Geshe Kelsang Gyatso's teachings about '{user_query}':\n\n"
    
    for i, (passage, source) in enumerate(zip(passages, sources)):
        # Truncate long passages
        if len(passage) > 300:
            passage = passage[:300] + "..."
            
        response += f"From {source}:\n"
        response += f"{passage}\n\n"
    
    response += "I hope these passages help answer your question. For more detailed information, you may want to consult the specific books mentioned above."
    
    return response

def check_processing_status():
    """Check if the EPUBs have been processed and a vector database exists"""
    logger.info("Checking processing status")
    
    # Check for vector database
    try:
        chroma_client = chromadb.PersistentClient(path=VECTORDB_DIR)
        collection = chroma_client.get_collection(name="geshe_kelsang_gyatso")
        # Get collection count to verify it has data
        count = collection.count()
        logger.info(f"Found existing vector database with {count} entries")
        if count > 0:
            return True, collection
        else:
            logger.warning("Vector database exists but is empty")
            return False, None
    except Exception as e:
        logger.warning(f"Vector database not found or error accessing it: {e}")
        return False, None

# %% [markdown]
# ## Interactive Explorer Interface
# This is the main user interface for the teachings explorer

# %%
# INTERACTIVE EXPLORER INTERFACE
# =============================

def run_teaching_explorer():
    """Main function to run the teachings explorer"""
    logger.info("Starting teaching explorer")
    
    # Display welcome header with quote
    display(HTML("""
    <style>
    .welcome-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .quote {
        font-style: italic;
        font-size: 1.2em;
        margin: 15px 0;
        padding: 10px;
        border-left: 4px solid white;
        display: inline-block;
    }
    .response-area {
        background-color: #f5f5f5;
        border-left: 3px solid #03A9F4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .citation {
        font-style: italic;
        color: #666;
        font-size: 0.9em;
    }
    .query-box {
        width: 100%;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 10px;
        font-family: Arial, sans-serif;
    }
    </style>
    <div class="welcome-header">
        <h1>Geshe Kelsang Gyatso Teachings Explorer</h1>
        <div class="quote">"I am my books" - Geshe Kelsang Gyatso</div>
    </div>
    """))
    
    # Check if we already have a user name
    global current_user_name
    if current_user_name:
        start_explorer_interface(current_user_name)
        return
    
    # Ask for user's name
    name_input = ipyw.Text(
        value='',
        placeholder='Please enter your name',
        description='Your Name:',
        disabled=False,
        layout=ipyw.Layout(width='50%')
    )
    
    name_button = ipyw.Button(
        description='Continue',
        button_style='primary',
        tooltip='Click to continue',
        layout=ipyw.Layout(width='200px')
    )
    
    name_output = ipyw.Output()
    
    # Function to handle name submission
    def on_name_submit(b):
        global current_user_name
        user_name = name_input.value.strip()
        if not user_name:
            user_name = "Friend"
        
        current_user_name = user_name
        
        with name_output:
            clear_output()
            # Start the main interface
            start_explorer_interface(user_name)
    
    # Connect button to handler
    name_button.on_click(on_name_submit)
    
    # Display name input
    display(HTML("<p>Before we begin, please tell me your name:</p>"))
    display(ipyw.HBox([name_input, name_button]))
    display(name_output)

def start_explorer_interface(user_name):
    """Start the main explorer interface after getting user's name"""
    logger.info(f"Starting explorer interface for user: {user_name}")
    
    # Check if processing is needed
    processing_status, collection = check_processing_status()
    
    # Setup anthropic client
    anthropic_client = setup_anthropic_client()
    
    # Welcome message
    display(HTML(f"""
    <h2>Welcome, {user_name}!</h2>
    <p>This explorer allows you to search and interact with the teachings of Geshe Kelsang Gyatso.</p>
    """))
    
    # If processing is needed, show processing interface
    if not processing_status:
        display(HTML("""
        <div style="padding: 15px; background-color: #fff3cd; border-left: 4px solid #ffc107; margin: 20px 0; border-radius: 5px;">
            <h3>⚠️ Teachings need to be processed</h3>
            <p>Before you can ask questions, I need to process the teachings from the EPUB files. This may take some time depending on how many books are available.</p>
        </div>
        """))
        
        process_button = ipyw.Button(
            description='Process Teachings',
            button_style='warning',
            tooltip='Click to process the teachings',
            icon='cogs'
        )
        
        process_output = ipyw.Output()
        
        # Function to handle processing
        def on_process_click(b):
            with process_output:
                clear_output()
                print("Processing teachings... This may take some time.")
                collection = process_all_epubs()
                clear_output()
                if isinstance(collection, str):
                    # Error message
                    display(HTML(f"""
                    <div style="padding: 15px; background-color: #f8d7da; border-left: 4px solid #dc3545; margin: 20px 0; border-radius: 5px;">
                        <h3>❌ Error</h3>
                        <p>{collection}</p>
                    </div>
                    """))
                else:
                    # Success message
                    display(HTML("""
                    <div style="padding: 15px; background-color: #d4edda; border-left: 4px solid #28a745; margin: 20px 0; border-radius: 5px;">
                        <h3>✅ Processing Complete</h3>
                        <p>The teachings have been processed successfully! You can now ask questions.</p>
                    </div>
                    """))
                    # Show the question interface
                    display_question_interface(collection, anthropic_client, user_name)
        
        # Connect button to handler
        process_button.on_click(on_process_click)
        
        # Display processing button
        display(process_button)
        display(process_output)
    else:
        # If already processed, show question interface directly
        display_question_interface(collection, anthropic_client, user_name)

def display_session_management(user_name):
    """Display session management controls"""
    # Session name input
    session_name = ipyw.Text(
        value='',
        placeholder='Session name (e.g., "Emptiness Study")',
        description='Save As:',
        layout=ipyw.Layout(width='300px')
    )
    
    save_btn = ipyw.Button(
        description='Save Session',
        button_style='info',
        icon='save',
        layout=ipyw.Layout(width='150px')
    )
    
    save_output = ipyw.Output()
    
    # Function to save session
    def on_save_session(b):
        with save_output:
            clear_output()
            name = session_name.value.strip()
            if not name:
                print("Please enter a session name")
                return
                
            result = save_session(name, user_name)
            print(result)
    
    save_btn.on_click(on_save_session)
    
    # Create session selector
    session_files = glob.glob(f"{SESSION_DIR}/*.json")
    session_options = [os.path.basename(f).replace('.json', '').replace('_', ' ') for f in session_files]
    
    session_dropdown = ipyw.Dropdown(
        options=[''] + session_options,
        description='Load:',
        layout=ipyw.Layout(width='300px')
    )
    
    load_btn = ipyw.Button(
        description='Load Session',
        button_style='info',
        icon='folder-open',
        layout=ipyw.Layout(width='150px')
    )
    
    load_output = ipyw.Output()
    
    # Function to load session
    def on_load_session(b):
        with load_output:
            clear_output()
            selected = session_dropdown.value
            if not selected:
                print("Please select a session")
                return
                
            result = load_session(selected)
            print(result)
            
            # If session loaded successfully, restart interface
            if "loaded successfully" in result:
                time.sleep(2)
                clear_output()
                run_teaching_explorer()
    
    load_btn.on_click(on_load_session)
    
    # Create the session management UI
    session_ui = ipyw.VBox([
        ipyw.HBox([session_name, save_btn]),
        save_output,
        ipyw.HBox([session_dropdown, load_btn]),
        load_output
    ])
    
    return session_ui

def display_question_interface(collection, anthropic_client, user_name):
    """Display the question interface"""
    logger.info(f"Displaying question interface for {user_name}")
    
    # Create input box
    query_box = ipyw.Textarea(
        placeholder='Enter your question about Geshe Kelsang Gyatso\'s teachings here...',
        layout=ipyw.Layout(width='100%', height='100px', margin='10px 0')
    )
    
    # Output area
    output_area = ipyw.Output(
        layout=ipyw.Layout(width='100%', min_height='200px')
    )
    
    # Submit button
    submit_btn = ipyw.Button(
        description='Ask Question',
        button_style='primary',
        tooltip='Click to submit your question',
        icon='search',
        layout=ipyw.Layout(width='200px')
    )
    
    # Simple mode toggle
    simple_toggle = ipyw.Checkbox(
        value=False,
        description='Simple response (fewer citations)',
        layout=ipyw.Layout(width='300px', margin='5px 15px')
    )
    
    # Memory toggle
    memory_toggle = ipyw.Checkbox(
        value=True,
        description='Use conversation memory',
        layout=ipyw.Layout(width='250px', margin='5px 15px')
    )
    
    # Status indicator
    status = ipyw.HTML()
    
    # Handle submission
    def on_submit(b):
        query = query_box.value.strip()
        simple_mode = simple_toggle.value
        use_memory = memory_toggle.value
        
        if not query:
            status.value = "<p style='color:red'>Please enter a question.</p>"
            return
        
        # Clear previous output and show status
        output_area.clear_output()
        status.value = "<p>Searching teachings and generating response...</p>"
        
        # Process the query
        try:
            # Get results
            with output_area:
                clear_output()
                print(f"Searching for: {query}")
                
                # Search vector database
                search_results = query_master_teachings(query, collection, top_k=5 if simple_mode else 8)
                
                print(f"Found {len(search_results['documents'][0])} relevant passages. Generating response...")
                
                # Generate response
                response = generate_response_with_claude(
                    query, search_results, anthropic_client, 
                    include_detailed_citations=not simple_mode,
                    user_name=user_name,
                    use_conversation_history=use_memory
                )
                
                # Save interaction to history
                interaction_id = save_interaction(user_name, query, response, search_results)
                
                # Extract topics
                topics = extract_topics(response, query)
                
                # Process response for display
                html_response = markdown.markdown(response)
                html_response = highlight_quotes_in_response(html_response)
                
                # Add feedback buttons
                feedback_html = add_feedback_buttons(interaction_id, query, response)
                
                # Add export option
                export_html = add_export_option(response, query)
                
                # Add topic tags
                topics_html = display_topics(topics)
                
                # Add similarity info
                similarity_html = display_similarity_info(search_results)
                
                # Display formatted response with all enhancements
                clear_output()
                display(HTML(f"""
                <div class='response-area'>
                    {html_response}
                    {topics_html}
                    {similarity_html}
                    {feedback_html}
                    {export_html}
                </div>
                """))
                
            status.value = ""
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            status.value = f"<p style='color:red'>Error: {str(e)}</p>"
    
    # Connect button to handler
    submit_btn.on_click(on_submit)
    
    # Create view history button
    history_btn = ipyw.Button(
        description='View History',
        button_style='info',
        tooltip='View past conversations',
        icon='history',
        layout=ipyw.Layout(width='200px')
    )
    
    # Function to display history
    def on_history_click(b):
        with output_area:
            clear_output()
            history_file = f"{HISTORY_DIR}/{user_name.lower().replace(' ', '_')}_history.json"
            
            if not os.path.exists(history_file):
                display(HTML("<p>No interaction history found.</p>"))
                return
            
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            if not history["interactions"]:
                display(HTML("<p>No interactions recorded yet.</p>"))
                return
            
            # Display history
            display(HTML("<h2>Interaction History</h2>"))
            
            for i, interaction in enumerate(reversed(history["interactions"][:20])):  # Show last 20
                display(HTML(f"""
                <div style='background-color:#f5f5f5; padding:10px; margin:10px 0; border-radius:5px;'>
                    <p><strong>Time:</strong> {interaction["timestamp"]} | <strong>User:</strong> {user_name}</p>
                    <p><strong>Question:</strong> {interaction["query"]}</p>
                    <details>
                        <summary>View Response</summary>
                        <div style='padding:10px;'>
                            {markdown.markdown(interaction["response"])}
                        </div>
                    </details>
                </div>
                """))
            
            # Add a note if there are more interactions
            if len(history["interactions"]) > 20:
                display(HTML(f"<p>Showing 20 most recent interactions of {len(history['interactions'])} total.</p>"))
    
    # Connect history button
    history_btn.on_click(on_history_click)
    
    # Create feedback analysis button
    feedback_btn = ipyw.Button(
        description='View Feedback Analysis',
        button_style='info',
        tooltip='View analysis of user feedback',
        icon='bar-chart',
        layout=ipyw.Layout(width='250px')
    )
    
    # Connect feedback analysis button
    feedback_btn.on_click(lambda b: display_feedback_analysis())
    
    # Create reset button
    reset_btn = ipyw.Button(
        description='New User',
        button_style='danger',
        tooltip='Reset for a new user',
        icon='refresh',
        layout=ipyw.Layout(width='150px')
    )
    
    # Connect reset button
    reset_btn.on_click(lambda b: reset_for_new_user())
    
    # Create session management
    session_mgmt = display_session_management(user_name)
    
    # Display all components
    display(query_box)
    display(ipyw.HBox([submit_btn, simple_toggle, memory_toggle, history_btn]))
    display(status)
    display(output_area)
    
    # Display session and system options in a collapsible section
    display(HTML("<hr><h3>Session & System Options:</h3>"))
    display(ipyw.VBox([
        ipyw.HBox([feedback_btn, reset_btn]),
        ipyw.HTML("<h4>Session Management:</h4>"),
        session_mgmt
    ]))
    
    # Instructions for users
    display(HTML("""
    <div style="margin-top: 20px; padding: 10px; background-color: #e8f5e9; border-radius: 5px;">
      <h3>How to use this explorer:</h3>
      <ol>
        <li>Type your question in the text box above</li>
        <li>Click "Ask Question" to submit</li>
        <li>The system will search through Geshe Kelsang Gyatso's teachings and provide a response</li>
        <li>For easier reading, check "Simple response" to get fewer citations</li>
        <li>To build on previous conversations, keep "Use conversation memory" checked</li>
        <li>Use "View History" to see your past interactions</li>
        <li>You can save your study sessions and load them later</li>
      </ol>
    </div>
    """))

# %% [markdown]
# ## Run the Explorer
# Run this cell to start using the Geshe Kelsang Gyatso Teachings Explorer

# %%
# RUN THE EXPLORER
# ===============

# Just run this cell to start the explorer
run_teaching_explorer()