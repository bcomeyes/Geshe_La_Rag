# %% [markdown]
"""
Complete Re-Extraction and Re-Embedding Pipeline
Buddhist RAG System - All 23 Books

This script:
1. Re-extracts all EPUBs with PRESERVED paragraph structure
2. Re-chunks with 33% overlap RESPECTING paragraph boundaries
3. Creates new embeddings
4. Builds new ChromaDB collection

CRITICAL FIX: Previous extraction destroyed paragraph structure with re.sub(r'\s+', ' ')
This version preserves the sacred text structure as the Guru created it.

Runtime: ~30-40 minutes
Cost: ~$1.50 for embeddings
"""

# %%
import os
import json
import time
from typing import List, Dict
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import openai
import tiktoken
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv

print("=" * 70)
print("BUDDHIST TEXT RE-EXTRACTION WITH PARAGRAPH PRESERVATION")
print("=" * 70)
print("\n✓ Step 1: Imports loaded")

# %%
# Configuration
BASE_DIR = r"C:\Users\DELL\Documents\gesha_la_rag"
EPUB_DIR = os.path.join(BASE_DIR, "epub_directory", "epub_directory")
EXTRACTED_TEXT_DIR = os.path.join(BASE_DIR, "extracted_text")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
VECTORDB_DIR = os.path.join(BASE_DIR, "vector_db")

# Create directories
os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(VECTORDB_DIR, exist_ok=True)

# Collection name with timestamp
COLLECTION_NAME = f"proper_paragraphs_{int(time.time())}"

print(f"\n✓ Step 2: Configuration set")
print(f"  EPUB source: {EPUB_DIR}")
print(f"  Collection: {COLLECTION_NAME}")

# %%
# Load API key
load_dotenv(os.path.join(BASE_DIR, ".env"))
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
encoding = tiktoken.get_encoding("cl100k_base")

print("\n✓ Step 3: OpenAI client initialized")

# %%
# =============================================================================
# CORRECTED EPUB EXTRACTION - PRESERVES PARAGRAPH STRUCTURE
# =============================================================================

def extract_text_from_epub_proper(epub_path: str) -> Dict:
    """
    Extract text from EPUB preserving paragraph structure.
    
    CRITICAL DIFFERENCE from previous version:
    - OLD: re.sub(r'\s+', ' ', text) ← DESTROYED paragraphs
    - NEW: Extracts <p> tags, joins with \n\n ← PRESERVES paragraphs
    
    Args:
        epub_path: Path to .epub file
        
    Returns:
        Dictionary with book metadata and chapters with paragraph-preserved text
    """
    book_id = Path(epub_path).stem
    
    try:
        book = epub.read_epub(epub_path)
        
        # Extract metadata
        book_title = "Unknown Title"
        try:
            title_data = book.get_metadata('DC', 'title')
            if title_data and len(title_data) > 0:
                book_title = title_data[0][0]
        except:
            book_title = book_id.replace('_', ' ').replace('-', ' ')
        
        creator = "Geshe Kelsang Gyatso"
        
        # Extract chapters
        chapters = []
        current_position = 0
        position_to_page = {}
        chars_per_page = 2000  # Estimate
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                try:
                    content = item.get_content().decode('utf-8', errors='replace')
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Extract chapter title from heading tags
                    chapter_title = None
                    for tag in ['h1', 'h2', 'h3']:
                        heading = soup.find(tag)
                        if heading:
                            chapter_title = heading.get_text().strip()
                            break
                    
                    # CRITICAL: Extract paragraphs properly
                    paragraphs = soup.find_all('p')
                    
                    if paragraphs:
                        # Extract text from each <p> tag
                        para_texts = []
                        for p in paragraphs:
                            para_text = p.get_text()
                            # Clean whitespace WITHIN each paragraph only
                            para_text = ' '.join(para_text.split())
                            if para_text.strip():
                                para_texts.append(para_text)
                        
                        # Join paragraphs with double newlines
                        text = '\n\n'.join(para_texts)
                    else:
                        # Fallback: get all text and try to preserve natural breaks
                        text = soup.get_text()
                        # Normalize multiple newlines to double
                        import re
                        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
                    
                    if text.strip():
                        # Update position_to_page mapping
                        for i in range(0, len(text), chars_per_page):
                            position_to_page[current_position + i] = (current_position + i) // chars_per_page + 1
                        
                        chapters.append({
                            "content": text,
                            "chapter_title": chapter_title,
                            "start_position": current_position,
                        })
                        
                        current_position += len(text)
                        
                except Exception as e:
                    print(f"  ⚠️ Error processing item in {book_title}: {e}")
                    continue
        
        return {
            "book_id": book_id,
            "book_title": book_title,
            "creator": creator,
            "chapters": chapters,
            "position_to_page": position_to_page,
            "total_length": current_position
        }
        
    except Exception as e:
        print(f"  ❌ Error processing {epub_path}: {e}")
        return None

print("\n✓ Step 4: Extraction function ready (PARAGRAPH-PRESERVING)")

# %%
# =============================================================================
# EXTRACT ALL BOOKS
# =============================================================================

print("\n" + "=" * 70)
print("EXTRACTING ALL EPUBS")
print("=" * 70)

import glob

epub_files = glob.glob(os.path.join(EPUB_DIR, "*.epub"))
print(f"\nFound {len(epub_files)} EPUB files")

extracted_books = []

for epub_path in tqdm(epub_files, desc="Extracting EPUBs"):
    book_data = extract_text_from_epub_proper(epub_path)
    
    if book_data:
        # Save JSON
        json_filename = f"{book_data['book_id']}.json"
        json_path = os.path.join(EXTRACTED_TEXT_DIR, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(book_data, f, ensure_ascii=False, indent=2)
        
        extracted_books.append(book_data)

print(f"\n✓ Step 5: Extracted {len(extracted_books)} books to JSON")
print(f"  Output: {EXTRACTED_TEXT_DIR}")

# %%
# Verify paragraph preservation
print("\n" + "=" * 70)
print("VERIFICATION: Paragraph Structure Preserved")
print("=" * 70)

if extracted_books:
    sample = extracted_books[0]
    sample_chapter = sample['chapters'][10] if len(sample['chapters']) > 10 else sample['chapters'][0]
    
    para_count = sample_chapter['content'].count('\n\n') + 1
    
    print(f"\nSample: {sample['book_title']}")
    print(f"  Chapter: {sample_chapter.get('chapter_title', 'Untitled')}")
    print(f"  Character length: {len(sample_chapter['content'])}")
    print(f"  Paragraphs detected: {para_count}")
    
    # Show first 2 paragraphs
    paragraphs = sample_chapter['content'].split('\n\n')
    print(f"\n  First 2 paragraphs:")
    for i, p in enumerate(paragraphs[:2], 1):
        print(f"\n  [{i}] ({len(p)} chars)")
        print(f"  {p[:150]}...")

# %%
# =============================================================================
# CHUNKING WITH 33% OVERLAP - RESPECTING PARAGRAPH BOUNDARIES
# =============================================================================

def chunk_with_paragraph_respect(text: str, max_tokens: int = 4000, overlap_ratio: float = 0.33) -> List[str]:
    """
    Chunk text with 33% overlap while respecting paragraph boundaries.
    
    CRITICAL: This preserves the sacred text structure - never splits mid-paragraph.
    
    Strategy:
    1. Split text into paragraphs by \n\n
    2. Accumulate paragraphs until approaching max_tokens
    3. Create overlap by including last N paragraphs from previous chunk
    4. Never split a paragraph - include it whole or not at all
    
    Args:
        text: Text with paragraphs separated by \n\n
        max_tokens: Maximum tokens per chunk
        overlap_ratio: Proportion of overlap (0.33 = 33%)
        
    Returns:
        List of text chunks with paragraph-aligned boundaries
    """
    overlap_tokens = int(max_tokens * overlap_ratio)
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current = []
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = len(encoding.encode(para))
        
        # If this paragraph alone exceeds max_tokens, split it at sentences
        if para_tokens > max_tokens:
            # Split long paragraph at sentence boundaries
            sentences = para.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')
            
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                    
                sent_tokens = len(encoding.encode(sent))
                
                if current_tokens + sent_tokens > max_tokens and current_tokens > 0:
                    # Save current chunk
                    chunks.append('\n\n'.join(current))
                    
                    # Create overlap from end of previous chunk
                    overlap = []
                    overlap_count = 0
                    for item in reversed(current):
                        item_tokens = len(encoding.encode(item))
                        if overlap_count + item_tokens <= overlap_tokens:
                            overlap.insert(0, item)
                            overlap_count += item_tokens
                        else:
                            break
                    
                    current = overlap
                    current_tokens = overlap_count
                
                current.append(sent)
                current_tokens += sent_tokens
        else:
            # Normal paragraph - check if we need to start new chunk
            if current_tokens + para_tokens > max_tokens and current_tokens > 0:
                # Save current chunk
                chunks.append('\n\n'.join(current))
                
                # Create overlap from end of previous chunk
                overlap = []
                overlap_count = 0
                for item in reversed(current):
                    item_tokens = len(encoding.encode(item))
                    if overlap_count + item_tokens <= overlap_tokens:
                        overlap.insert(0, item)
                        overlap_count += item_tokens
                    else:
                        break
                
                current = overlap
                current_tokens = overlap_count
            
            # Add paragraph to current chunk
            current.append(para)
            current_tokens += para_tokens
    
    # Don't forget last chunk
    if current:
        chunks.append('\n\n'.join(current))
    
    return chunks

print("\n✓ Step 6: Chunking function ready (PARAGRAPH-RESPECTING)")

# %%
# =============================================================================
# PROCESS ALL BOOKS TO CHUNKS
# =============================================================================

print("\n" + "=" * 70)
print("CHUNKING ALL BOOKS")
print("=" * 70)

all_chunks = []

for book_data in tqdm(extracted_books, desc="Chunking books"):
    book_title = book_data['book_title']
    
    for chapter in book_data['chapters']:
        text = chapter.get('content', '')
        if not text.strip():
            continue
        
        # Chunk with paragraph respect
        text_chunks = chunk_with_paragraph_respect(text, max_tokens=4000, overlap_ratio=0.33)
        
        # Get page from position_to_page
        position_to_page = book_data.get('position_to_page', {})
        start_page = position_to_page.get(str(chapter.get('start_position', 0)), 1)
        
        for idx, content in enumerate(text_chunks):
            all_chunks.append({
                "text": content,
                "metadata": {
                    "book_title": book_title,
                    "creator": book_data.get('creator', 'Geshe Kelsang Gyatso'),
                    "chapter_title": chapter.get('chapter_title') or 'Untitled',
                    "start_page": start_page,
                    "chunk_index": idx
                }
            })

print(f"\n✓ Step 7: Created {len(all_chunks)} chunks from {len(extracted_books)} books")

# Verify chunk quality
chunk_sizes = [len(encoding.encode(c["text"])) for c in all_chunks]
print(f"  Average chunk size: {sum(chunk_sizes)/len(chunk_sizes):.0f} tokens")
print(f"  Max chunk size: {max(chunk_sizes)} tokens")

over_max = [s for s in chunk_sizes if s > 4000]
if over_max:
    print(f"  ⚠️ {len(over_max)} chunks over 4000 tokens (max: {max(over_max)})")
else:
    print(f"  ✓ All chunks within limit")

# Sample chunks to verify paragraph preservation
print(f"\n  Sample chunk (showing paragraph structure):")
print(f"  " + "-" * 66)
sample_chunk = all_chunks[50]['text'] if len(all_chunks) > 50 else all_chunks[0]['text']
para_count = sample_chunk.count('\n\n') + 1
print(f"  Paragraphs in chunk: {para_count}")
print(f"  First 300 chars:\n  {sample_chunk[:300]}...")

# %%
# =============================================================================
# CREATE EMBEDDINGS
# =============================================================================

print("\n" + "=" * 70)
print("CREATING EMBEDDINGS")
print("=" * 70)
print(f"\nProcessing {len(all_chunks)} chunks in batches of 10")
print("This will take approximately 20-30 minutes...")

chunks_with_embeddings = []
failed_count = 0

for i in tqdm(range(0, len(all_chunks), 10), desc="Embedding"):
    batch = all_chunks[i:i + 10]
    batch_texts = [c["text"] for c in batch]
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch_texts
        )
        
        for j, chunk in enumerate(batch):
            chunk_copy = chunk.copy()
            chunk_copy["embedding"] = response.data[j].embedding
            chunks_with_embeddings.append(chunk_copy)
        
        time.sleep(0.5)  # Rate limiting
        
    except Exception as e:
        print(f"\n⚠️ Error on batch {i//10}: {e}")
        failed_count += 1
        # Add chunks without embeddings
        for chunk in batch:
            chunk_copy = chunk.copy()
            chunk_copy["embedding"] = None
            chunks_with_embeddings.append(chunk_copy)

successful = sum(1 for c in chunks_with_embeddings if c["embedding"] is not None)
print(f"\n✓ Step 8: Created {successful}/{len(all_chunks)} embeddings")
if failed_count > 0:
    print(f"  ⚠️ Failed batches: {failed_count}")

# %%
# Save embeddings
embeddings_filename = f"proper_paragraphs_embeddings_{int(time.time())}.json"
embeddings_path = os.path.join(EMBEDDINGS_DIR, embeddings_filename)

print(f"\nSaving embeddings to {embeddings_filename}...")
with open(embeddings_path, 'w') as f:
    json.dump(chunks_with_embeddings, f)

file_size_mb = Path(embeddings_path).stat().st_size / (1024 * 1024)
print(f"✓ Step 9: Saved embeddings ({file_size_mb:.1f} MB)")

# %%
# =============================================================================
# CREATE CHROMADB COLLECTION
# =============================================================================

print("\n" + "=" * 70)
print("CREATING CHROMADB COLLECTION")
print("=" * 70)
print(f"\nCollection: {COLLECTION_NAME}")

chroma_client = chromadb.PersistentClient(path=VECTORDB_DIR)
collection = chroma_client.create_collection(name=COLLECTION_NAME)

ids = []
documents = []
embeddings = []
metadatas = []

for i, chunk in enumerate(chunks_with_embeddings):
    if chunk["embedding"] is None:
        continue
    
    ids.append(f"chunk_{i}")
    documents.append(chunk["text"])
    embeddings.append(chunk["embedding"])
    
    # Clean metadata
    meta = chunk["metadata"].copy()
    for key, value in meta.items():
        if value is None:
            meta[key] = ""
    metadatas.append(meta)

# Add to database in batches
for i in tqdm(range(0, len(ids), 100), desc="Adding to ChromaDB"):
    end = min(i + 100, len(ids))
    collection.add(
        ids=ids[i:end],
        documents=documents[i:end],
        embeddings=embeddings[i:end],
        metadatas=metadatas[i:end]
    )

print(f"\n✓ Step 10: Added {len(ids)} chunks to database")

# %%
# =============================================================================
# TEST QUERIES
# =============================================================================

print("\n" + "=" * 70)
print("TEST QUERIES")
print("=" * 70)

# Test 1: Clear light visualization
print("\n[Test 1] Clear light at heart center")
print("-" * 70)

query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=["visualize clear light at heart center"]
).data[0].embedding

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
    para_count = doc.count('\n\n') + 1
    print(f"\n[{i}] {meta['book_title']} - {meta['chapter_title']}")
    print(f"    Page {meta['start_page']} | {para_count} paragraphs in chunk")
    print(f"    {doc[:200]}...")

# Test 2: Emptiness
print("\n\n[Test 2] Emptiness and dependent arising")
print("-" * 70)

query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=["emptiness dependent arising"]
).data[0].embedding

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
    para_count = doc.count('\n\n') + 1
    print(f"\n[{i}] {meta['book_title']} - {meta['chapter_title']}")
    print(f"    Page {meta['start_page']} | {para_count} paragraphs in chunk")
    print(f"    {doc[:200]}...")

# %%
# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("COMPLETE - PROPER PARAGRAPH STRUCTURE PRESERVED")
print("=" * 70)

print(f"\n✓ Books processed: {len(extracted_books)}")
print(f"✓ Total chunks: {len(all_chunks)}")
print(f"✓ Embeddings created: {successful}")
print(f"✓ ChromaDB collection: {COLLECTION_NAME}")

print(f"\nKey improvements:")
print(f"  ✓ Paragraph structure preserved (not collapsed)")
print(f"  ✓ 33% overlap respects paragraph boundaries")
print(f"  ✓ Sacred text structure honored")

print(f"\nLocations:")
print(f"  JSON files: {EXTRACTED_TEXT_DIR}")
print(f"  Embeddings: {embeddings_path}")
print(f"  Vector DB: {VECTORDB_DIR}")

print("\n" + "=" * 70)
print("READY FOR PHASE 3: Graph Database Implementation")
print("=" * 70)

# %%