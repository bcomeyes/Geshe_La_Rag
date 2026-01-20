# %% [markdown]
# # Clear Light of Bliss - Vector Database
# 
# **INSTRUCTIONS: Run ALL cells in order from top to bottom**
# Runtime: ~3 minutes

# %%
# Imports
import os
import json
import re
import time
from typing import List, Dict

import openai
import tiktoken
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv

print("✅ Step 1: Imports loaded")

# %%
# Configuration
BASE_DIR = r"C:\Users\DELL\Documents\gesha_la_rag"
EXTRACTED_TEXT_DIR = os.path.join(BASE_DIR, "extracted_text")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
VECTORDB_DIR = os.path.join(BASE_DIR, "vector_db")

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(VECTORDB_DIR, exist_ok=True)

# Use unique collection name (avoids conflicts)
COLLECTION_NAME = f"clear_light_{int(time.time())}"

print(f"✅ Step 2: Configuration set")
print(f"   Collection: {COLLECTION_NAME}")

# %%
# Load API key
load_dotenv(os.path.join(BASE_DIR, ".env"))
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
encoding = tiktoken.get_encoding("cl100k_base")

print("✅ Step 3: API key loaded")

# %%
# Chunking function (handles long paragraphs)
def chunk_text(text: str, max_tokens: int = 4000, overlap_ratio: float = 0.33) -> List[str]:
    overlap_tokens = int(max_tokens * overlap_ratio)
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current = []
    current_tokens = 0
    
    for para in paragraphs:
        if not para.strip():
            continue
        
        para_tokens = len(encoding.encode(para))
        
        # Split long paragraphs at sentences
        if para_tokens > max_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_tokens = len(encoding.encode(sent))
                
                if current_tokens + sent_tokens > max_tokens and current_tokens > 0:
                    chunks.append("\n\n".join(current))
                    
                    # Create overlap
                    overlap = []
                    overlap_count = 0
                    for p in reversed(current):
                        p_tok = len(encoding.encode(p))
                        if overlap_count + p_tok <= overlap_tokens:
                            overlap.insert(0, p)
                            overlap_count += p_tok
                        else:
                            break
                    
                    current = overlap
                    current_tokens = overlap_count
                
                current.append(sent)
                current_tokens += sent_tokens
        else:
            if current_tokens + para_tokens > max_tokens and current_tokens > 0:
                chunks.append("\n\n".join(current))
                
                # Create overlap
                overlap = []
                overlap_count = 0
                for p in reversed(current):
                    p_tok = len(encoding.encode(p))
                    if overlap_count + p_tok <= overlap_tokens:
                        overlap.insert(0, p)
                        overlap_count += p_tok
                    else:
                        break
                
                current = overlap
                current_tokens = overlap_count
            
            current.append(para)
            current_tokens += para_tokens
    
    if current:
        chunks.append("\n\n".join(current))
    
    return chunks

print("✅ Step 4: Chunking function ready")

# %%
# Load Clear Light of Bliss
clb_path = os.path.join(EXTRACTED_TEXT_DIR, "Clear_Light_of_Bliss.json")

with open(clb_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ Step 5: Loaded {data['book_title']}")

# %%
# Process to chunks
all_chunks = []

for chapter in data['chapters']:
    text = chapter.get('content', '')
    if not text.strip():
        continue
    
    chunks = chunk_text(text, max_tokens=4000, overlap_ratio=0.33)
    
    position_to_page = data.get('position_to_page', {})
    start_page = position_to_page.get(str(chapter.get('start_position', 0)), 1)
    
    for idx, chunk_text in enumerate(chunks):
        all_chunks.append({
            "text": chunk_text,
            "metadata": {
                "book_title": data['book_title'],
                "creator": data['creator'],
                "chapter_title": chapter.get('chapter_title') or 'Untitled',
                "start_page": start_page,
                "chunk_index": idx
            }
        })

print(f"✅ Step 6: Created {len(all_chunks)} chunks")

# Verify chunk sizes
chunk_sizes = [len(encoding.encode(c["text"])) for c in all_chunks]
print(f"   Avg size: {sum(chunk_sizes)/len(chunk_sizes):.0f} tokens")
print(f"   Max size: {max(chunk_sizes)} tokens")

# %%
# Create embeddings
print("Creating embeddings (this takes ~2 minutes)...")

chunks_with_embeddings = []

for i in tqdm(range(0, len(all_chunks), 10)):
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
        
        time.sleep(0.5)
        
    except Exception as e:
        print(f"\nError: {e}")
        for chunk in batch:
            chunk_copy = chunk.copy()
            chunk_copy["embedding"] = None
            chunks_with_embeddings.append(chunk_copy)

successful = sum(1 for c in chunks_with_embeddings if c["embedding"] is not None)
print(f"\n✅ Step 7: Created {successful}/{len(all_chunks)} embeddings")

# %%
# Save embeddings
embeddings_path = os.path.join(EMBEDDINGS_DIR, "clear_light_embeddings.json")
with open(embeddings_path, 'w') as f:
    json.dump(chunks_with_embeddings, f)

print(f"✅ Step 8: Saved embeddings")

# %%
# Create ChromaDB collection
print(f"Creating collection: {COLLECTION_NAME}")

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

# Add to database
for i in tqdm(range(0, len(ids), 100), desc="Adding to ChromaDB"):
    end = min(i + 100, len(ids))
    collection.add(
        ids=ids[i:end],
        documents=documents[i:end],
        embeddings=embeddings[i:end],
        metadatas=metadatas[i:end]
    )

print(f"✅ Step 9: Added {len(ids)} chunks to database")

# %%
# Test query 1
print("\nTest Query 1:")
print("="*70)
results = collection.query(
    query_texts=["visualize clear light at heart center"],
    n_results=3
)

for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
    print(f"\n[{i}] {meta['chapter_title']} (Page {meta['start_page']})")
    print(f"    {doc[:200]}...")

# %%
# Test query 2
print("\nTest Query 2:")
print("="*70)
results = collection.query(
    query_texts=["emptiness and bliss relationship"],
    n_results=3
)

for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
    print(f"\n[{i}] {meta['chapter_title']} (Page {meta['start_page']})")
    print(f"    {doc[:200]}...")

# %%
# Summary
print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"Collection: {COLLECTION_NAME}")
print(f"Chunks: {len(all_chunks)}")
print(f"Embeddings: {successful}")
print(f"Location: {VECTORDB_DIR}")
print("="*70)
print("\n✅ Ready for Phase 2 and Phase 3")

# %%