# %%
"""
Phase 3: Neo4j Graph Database Implementation
Buddhist RAG System - Clear Light of Bliss

This notebook implements the dual-layer graph architecture:
- Layer 1: Document Structure (Book → Chapter → Page → Paragraph)
- Layer 2: Semantic Concepts (Extracted relationships with source provenance)

Author: Matt's Buddhist RAG Project
Date: January 2026
"""

# %%
# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# For Phase 2 NLP (already working)
import spacy
from spacy.language import Language

# For Neo4j (will install later)
# from neo4j import GraphDatabase

print("Imports successful!")
print(f"Python environment ready for Phase 3 implementation")

# %%
# =============================================================================
# DATA CLASSES FOR STRUCTURED STORAGE
# =============================================================================
# These classes define the structure of our parsed document hierarchy
# Using dataclasses makes the code cleaner and provides automatic __init__ methods

@dataclass
class ParagraphMetadata:
    """
    Represents a single paragraph with its location in the document.
    
    This is the atomic unit of text that connects Layer 1 (structure) 
    to Layer 2 (semantic relationships).
    """
    paragraph_id: str          # Unique identifier: "clb_ch{chapter_idx}_para{para_idx}"
    text: str                  # The actual paragraph text
    chapter_index: int         # Which chapter (0-based)
    paragraph_index: int       # Position within chapter (0-based)
    char_start: int           # Character offset from book start
    char_end: int             # Character offset from book start
    page_number: int          # Page number this paragraph appears on
    sentence_count: int       # Number of sentences (for metadata)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class ChapterMetadata:
    """
    Represents a chapter with all its paragraphs.
    
    Note: Some 'chapters' are actually front matter (title page, copyright, etc.)
    with chapter_title = None. We'll handle these appropriately.
    """
    chapter_index: int
    chapter_title: Optional[str]  # None for front matter
    start_position: int           # Character offset where chapter begins
    pages: List[int]              # All page numbers this chapter spans
    paragraphs: List[ParagraphMetadata]
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'chapter_index': self.chapter_index,
            'chapter_title': self.chapter_title,
            'start_position': self.start_position,
            'pages': self.pages,
            'paragraphs': [p.to_dict() for p in self.paragraphs]
        }

@dataclass  
class DocumentStructure:
    """
    Complete hierarchical structure of the book.
    
    This represents Layer 1 of our graph database.
    """
    book_id: str
    book_title: str
    creator: str
    total_chapters: int
    total_paragraphs: int
    total_pages: int
    chapters: List[ChapterMetadata]
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'book_id': self.book_id,
            'book_title': self.book_title,
            'creator': self.creator,
            'total_chapters': self.total_chapters,
            'total_paragraphs': self.total_paragraphs,
            'total_pages': self.total_pages,
            'chapters': [c.to_dict() for c in self.chapters]
        }

print("Data structures defined successfully!")
print("\nKey design decision: Using dataclasses for type safety and clean code")
print("Each paragraph gets a unique ID: 'clb_ch{chapter}_para{paragraph}'")

# %%
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def count_sentences(text: str) -> int:
    """
    Count sentences in a paragraph using simple heuristics.
    
    This is approximate - we use sentence-ending punctuation followed by space or newline.
    Buddhist texts may have complex punctuation (citations, Sanskrit terms, etc.),
    so this is a rough estimate for metadata purposes.
    
    Args:
        text: Paragraph text to analyze
        
    Returns:
        Approximate sentence count
    """
    # Simple sentence detection: period/exclamation/question followed by space/newline or end of string
    sentence_endings = re.findall(r'[.!?](?:\s|$)', text)
    return max(1, len(sentence_endings))  # Minimum 1 sentence per paragraph

def find_page_number(char_position: int, position_to_page: Dict[str, int]) -> int:
    """
    Find which page a character position falls on.
    
    The position_to_page dictionary has keys as strings (character positions) 
    and values as page numbers. We need to find the largest position that's 
    still <= our target position.
    
    How this works:
    - position_to_page = {"0": 1, "2518": 2, "3368": 2, "4009": 3, ...}
    - If char_position = 3500, we want page 2 (because 3368 <= 3500 < 4009)
    
    Args:
        char_position: Character offset to look up
        position_to_page: Dictionary mapping position strings to page numbers
        
    Returns:
        Page number where this position appears
    """
    # Convert string keys to integers and sort them
    positions = sorted([int(pos) for pos in position_to_page.keys()])
    
    # Find the largest position <= char_position
    page = 1  # Default to page 1 if something goes wrong
    for pos in positions:
        if pos <= char_position:
            page = position_to_page[str(pos)]
        else:
            break  # We've gone past our target position
            
    return page

def extract_pages_for_chapter(start_pos: int, end_pos: int, position_to_page: Dict[str, int]) -> List[int]:
    """
    Determine which pages a chapter spans.
    
    A chapter might start on page 23 and end on page 45, so we need all pages in between.
    
    Args:
        start_pos: Character position where chapter starts
        end_pos: Character position where chapter ends
        position_to_page: Page mapping dictionary
        
    Returns:
        Sorted list of unique page numbers this chapter spans
    """
    start_page = find_page_number(start_pos, position_to_page)
    end_page = find_page_number(end_pos, position_to_page)
    
    # Return all pages from start to end (inclusive)
    return list(range(start_page, end_page + 1))

print("Helper functions defined!")
print("\nKey implementation notes:")
print("- Sentence counting is approximate (good enough for metadata)")
print("- Page lookup uses binary search concept (find largest position <= target)")
print("- Chapter pages calculated from start/end positions")

# %%
# =============================================================================
# DOCUMENT STRUCTURE PARSER - CORE FUNCTION
# =============================================================================

def parse_document_structure(json_path: str) -> DocumentStructure:
    """
    Parse Buddhist text JSON into hierarchical Book → Chapter → Page → Paragraph structure.
    
    This is the foundation of Layer 1 in our graph database. We parse the physical
    structure of the document so we can later attach semantic relationships to 
    specific locations.
    
    Key parsing rules:
    1. Paragraphs detected by \\n\\n breaks (double newline)
    2. Empty paragraphs filtered out
    3. Character positions tracked relative to book start
    4. Page numbers derived from position_to_page mapping
    5. Each paragraph gets unique ID: "clb_ch{chapter_idx}_para{para_idx}"
    
    Args:
        json_path: Path to Clear_Light_of_Bliss.json
        
    Returns:
        DocumentStructure with complete hierarchy
    """
    print(f"Loading JSON from: {json_path}")
    
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded book: {data['book_title']}")
    print(f"✓ Total chapters in JSON: {len(data['chapters'])}")
    
    # Extract top-level metadata
    book_id = data['book_id']
    book_title = data['book_title']
    creator = data['creator']
    position_to_page = data['position_to_page']
    
    # Get max page number
    max_page = max(position_to_page.values())
    print(f"✓ Book spans {max_page} pages")
    
    # Parse each chapter
    chapters = []
    total_paragraphs = 0
    
    for chapter_idx, chapter_data in enumerate(data['chapters']):
        chapter_title = chapter_data['chapter_title']
        chapter_content = chapter_data['content']
        chapter_start = chapter_data['start_position']
        
        # Calculate chapter end position
        # (It's the start of the next chapter, or end of book)
        if chapter_idx < len(data['chapters']) - 1:
            chapter_end = data['chapters'][chapter_idx + 1]['start_position']
        else:
            chapter_end = data['total_length']
        
        # Determine which pages this chapter spans
        chapter_pages = extract_pages_for_chapter(
            chapter_start, 
            chapter_end, 
            position_to_page
        )
        
        # Split content into paragraphs by double newline
        # We use \\n\\n as the paragraph delimiter per the project spec
        raw_paragraphs = chapter_content.split('\\n\\n')
        
        # Process each paragraph
        paragraphs = []
        current_position = chapter_start  # Track position as we process
        
        for para_idx, para_text in enumerate(raw_paragraphs):
            # Strip whitespace
            para_text = para_text.strip()
            
            # Skip empty paragraphs
            if not para_text:
                continue
            
            # Calculate character positions
            char_start = current_position
            char_end = current_position + len(para_text)
            
            # Find page number
            page_num = find_page_number(char_start, position_to_page)
            
            # Count sentences
            sent_count = count_sentences(para_text)
            
            # Create unique paragraph ID
            paragraph_id = f"clb_ch{chapter_idx}_para{para_idx}"
            
            # Create paragraph metadata object
            para_meta = ParagraphMetadata(
                paragraph_id=paragraph_id,
                text=para_text,
                chapter_index=chapter_idx,
                paragraph_index=para_idx,
                char_start=char_start,
                char_end=char_end,
                page_number=page_num,
                sentence_count=sent_count
            )
            
            paragraphs.append(para_meta)
            
            # Update position for next paragraph
            # Add length of text + \\n\\n separator (4 chars)
            current_position = char_end + 4
        
        # Create chapter metadata
        chapter_meta = ChapterMetadata(
            chapter_index=chapter_idx,
            chapter_title=chapter_title,
            start_position=chapter_start,
            pages=chapter_pages,
            paragraphs=paragraphs
        )
        
        chapters.append(chapter_meta)
        total_paragraphs += len(paragraphs)
        
        # Progress update
        title_display = chapter_title if chapter_title else f"[Front Matter {chapter_idx}]"
        print(f"  Chapter {chapter_idx}: {title_display}")
        print(f"    └─ {len(paragraphs)} paragraphs, pages {min(chapter_pages)}-{max(chapter_pages)}")
    
    # Create final document structure
    doc_structure = DocumentStructure(
        book_id=book_id,
        book_title=book_title,
        creator=creator,
        total_chapters=len(chapters),
        total_paragraphs=total_paragraphs,
        total_pages=max_page,
        chapters=chapters
    )
    
    print(f"\n{'='*70}")
    print(f"PARSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total chapters: {doc_structure.total_chapters}")
    print(f"Total paragraphs: {doc_structure.total_paragraphs}")
    print(f"Total pages: {doc_structure.total_pages}")
    print(f"{'='*70}")
    
    return doc_structure

print("Document structure parser ready!")
print("\nThis function is the heart of Layer 1 - it creates the physical hierarchy")
print("that will anchor all our semantic relationships in the graph database.")

# %%
# =============================================================================
# TEST THE PARSER - EXPLORE THE STRUCTURE
# =============================================================================

# Path to the JSON file
JSON_PATH = os.path.expanduser("~/Documents/gesha_la_rag/extracted_text/Clear_Light_of_Bliss.json")

# Parse the document
print("PARSING CLEAR LIGHT OF BLISS")
print("="*70)
doc_structure = parse_document_structure(JSON_PATH)

# %%
# Explore a sample chapter - let's look at a teaching chapter (not front matter)
print("\nEXPLORING SAMPLE CHAPTER")
print("="*70)

# Find first chapter with actual title (teaching content)
teaching_chapters = [ch for ch in doc_structure.chapters if ch.chapter_title is not None]

if teaching_chapters:
    sample_chapter = teaching_chapters[0]
    print(f"Chapter: {sample_chapter.chapter_title}")
    print(f"Index: {sample_chapter.chapter_index}")
    print(f"Pages: {min(sample_chapter.pages)} - {max(sample_chapter.pages)}")
    print(f"Total paragraphs: {len(sample_chapter.paragraphs)}")
    print(f"\nFirst 3 paragraphs:")
    print("-" * 70)
    
    for i, para in enumerate(sample_chapter.paragraphs[:3]):
        print(f"\nParagraph {i} (ID: {para.paragraph_id})")
        print(f"Page: {para.page_number}, Sentences: {para.sentence_count}")
        print(f"Chars: {para.char_start}-{para.char_end}")
        print(f"Text preview: {para.text[:200]}...")

# %%
# =============================================================================
# SAVE DOCUMENT STRUCTURE TO JSON
# =============================================================================

def save_document_structure(doc_structure: DocumentStructure, output_path: str):
    """
    Save the parsed document structure to JSON file.
    
    This creates a standalone representation of Layer 1 that can be:
    - Loaded quickly without re-parsing
    - Inspected manually
    - Used as input for Neo4j population
    
    Args:
        doc_structure: Parsed document structure
        output_path: Where to save the JSON file
    """
    output_dict = doc_structure.to_dict()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✓ Saved document structure to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")

# Save the structure
CHECKPOINT_DIR = Path(os.path.expanduser("~/Documents/gesha_la_rag/checkpoints"))
CHECKPOINT_DIR.mkdir(exist_ok=True)

output_path = CHECKPOINT_DIR / "06_document_structure_layer1.json"
save_document_structure(doc_structure, str(output_path))

print("\n" + "="*70)
print("PHASE 3 STEP 1 COMPLETE: Document Structure Parsed")
print("="*70)
print(f"Next steps:")
print(f"1. Load Phase 2 NLP model (EntityRuler + extraction function)")
print(f"2. Add source metadata to relationship extractions")
print(f"3. Extract from full book with progress tracking")
print(f"4. Set up Neo4j database")
print("="*70)

# %%