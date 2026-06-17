# Buddhist RAG System: Complete NLP Pipeline Reference (v2.1)

## Lessons Learned Edition

**Key Insight**: This document now includes **validation gates** and **exploratory analysis steps** that were missing in v1, leading to repeated rework. Each gate must pass before proceeding.

---

## Revision Log

- **v2.1 (June 17, 2026)** — Added "Known Issues & Deferred Work" section
  capturing KI-001 (lost word-boundary spaces), KI-002 (noisy relationship
  vocabulary), KI-003 (disconnected sub-graphs). Added Phase 3.4 "Graph
  Refinement" between Phase 3 and Phase 4. Updated Phase 4 with prerequisites
  and known-limitations notes. Updated Show Me tests and Complete Checklist.
  All additions originate from the June 17 Neo4j Browser exploration session.
- **v2.0 (January 2026)** — Lessons Learned Edition; validation gates added.
- **v1.0** — Initial pipeline reference.

---

## Project Overview

**Purpose**: Build a sophisticated RAG system for Geshe Kelsang Gyatso's teachings with dual modes:
1. **Academic Research**: Cross-book concept exploration with precise citations
2. **Guided Meditation**: Natural pause generation respecting instruction boundaries

**Scope**: Starting with "Clear Light of Bliss" (102k tokens) as proof of concept, scaling to 23 English books, eventually 675,000 pages of Tibetan texts.

**Core Principle**: The actual words as created by the Guru are sacred. All technical decisions prioritize preservation of teaching structure and integrity.

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Vector DB | ChromaDB | Semantic similarity search |
| Embeddings | OpenAI text-embedding-3-small (384 dim) | Dense vector encoding |
| NLP | spaCy en_core_web_lg + custom EntityRuler | NER + dependency parsing |
| Graph DB | Neo4j | Relationship traversal + provenance |
| Integration | Python orchestration | Dual-mode query routing |

---

## Known Issues & Deferred Work

**⚠️ Standing inventory of corpus-quality issues discovered during exploration.**
**Items here are deferred to specific later phases — NOT forgotten.**

### How this section works

Every entry has:
- **ID** (KI-NNN) for cross-referencing across the doc and chat sessions
- **Status** — current state; updated as items move through the pipeline
- **Discovered** — where/when we first noticed the issue (so it's traceable)
- **Examples** — concrete data points (so we have test cases later)
- **Trigger condition** — what causes us to actually address this
- **Rationale for deferral** — why it's not being addressed now

When an item is resolved, update its status to `RESOLVED in Phase X.Y` with a
link to the resolving notebook. Do not remove it from this section.

---

### KI-001: Lost word-boundary spaces in paragraph text

**Status**: DEFERRED — to multi-book extractor rebuild

**Discovered**: June 17, 2026 exploration session. While reading paragraphs
`CLB.6.p1` and `CLB.6.p2` directly in the Neo4j Browser, systematic missing
spaces at word boundaries were observed.

**Root cause**: EPUB-to-text extraction (Phase 1.1) stripped HTML tags around
italicized or styled inline text without preserving the implicit whitespace
those tags carried. Words flanking formatting markup got concatenated.

**Examples** (test cases for the eventual fix):

| Observed | Should be |
|---|---|
| `Tsongkhapa'sLamp` | `Tsongkhapa's Lamp` |
| `commentary to theSix Yogas` | `commentary to the Six Yogas` |
| `Mahamudra,TheMain Path` | `Mahamudra, The Main Path` |
| `autocommentary,Lamp of Re-illumination` | `autocommentary, Lamp of Re-illumination` |
| `ofMahamudra` | `of Mahamudra` |
| `asJoyful Path` | `as Joyful Path` |
| `variouspreliminaries` | `various preliminaries` |
| `a qualifiedVajrayana` | `a qualified Vajrayana` |
| `Tantraempowerment` | `Tantra empowerment` |
| `ourvows andcommitments` | `our vows and commitments` |
| `withfaith` | `with faith` |

**Pattern**: lowercase-letter immediately followed by uppercase-letter (`sL`,
`oM`, `tT`), or punctuation immediately followed by letter (`,T`, `.A`), are
the most common indicators. **NOT always** — proper compounds (`MacBook`,
`iPhone`) match the same pattern, so any regex needs a denylist.

**Trigger condition**: Address when extending extraction to a second book.
The cheapest fix is upstream in the extractor itself — re-running a fixed
extractor over all 26 books is one job, while running 26 separate downstream
regex passes is 26 jobs each carrying its own risk of introducing errors.

**What this means for Phase 4 results**: Quoted passages in Claude's responses
will preserve these surface errors. This is acceptable as a temporary quality
limitation. Add a note to Phase 4 output disclosing the data caveat until
KI-001 is resolved.

**Rationale for deferral**:
- Single-book regex pass would need to run again for every new book
- Fixing in the extractor itself is a one-time cost
- Phase 4 retrieval embeddings are robust to small surface noise
- We have concrete test cases captured above; nothing is being forgotten

---

### KI-002: Noisy doctrinal-relationship vocabulary

**Status**: SCHEDULED for Phase 3.4b

**Discovered**: June 17, 2026 exploration session. Running a relationship-type
profile around `clear_light` returned: `OF: 71`, `DEPENDS_UPON: 10`,
`KNOWN_AS: 6`, `AND: 4`, `MEDITATING_ON: 3`, `ARISE_FROM: 3`, `DISSOLVE_WITHIN: 2`,
`ENGAGING_IN: 1`, `ON: 1`, `FREE_FROM: 1`.

The most common relationship type for the most-mentioned tantric concept is
the English preposition `OF`.

**Root cause**: Phase 2/3 relation extractor used grammatical surface markers
(prepositions, conjunctions) as relationship types instead of either filtering
them or aggregating them under a non-doctrinal label.

**Confirmation from source data**: `07_semantic_relationships.json` metadata
explicitly classifies 311 of 683 relationships as `PREPOSITION` and 125 as
`CONJUNCTION`. The pipeline knew these were grammatical at extraction time —
but the surface label was preserved when ingested into Neo4j.

**Resolution**: Phase 3.4b. Relabel non-doctrinal relationship types as
`CO_OCCURS_WITH`, preserving the co-occurrence signal but removing the false
implication of doctrinal meaning. Doctrinal verbs unchanged.

---

### KI-003: Concept layer disconnected from Paragraph layer

**Status**: SCHEDULED for Phase 3.4a (in progress)

**Discovered**: June 17, 2026 exploration session. The query
`MATCH (c:Concept {canonical_form: 'clear_light'})-[]-(p:Paragraph)` returned
zero results. No edge type exists between Concept and Paragraph nodes in the
graph as populated by Phase 3.3.

**Root cause**: Phase 3.3 created two functional sub-graphs (the
Book→Chapter→Paragraph structural tree and the Concept↔Concept doctrinal web)
but never built the bridge edges connecting them. The data for the bridge
exists — every relationship in `07_semantic_relationships.json` has a
`source.citation` field — but it was used to set `mention_count` properties
on Concept nodes instead of producing edges.

**Resolution**: Phase 3.4a. Build `MENTIONS` edges from each Concept to every
Paragraph where it appears, using the `source.citation` field as the join key.

---

## Target Schema (DEFINE FIRST!)

**âš ï¸ LESSON LEARNED**: We never defined what the final data must look like. Define this BEFORE writing any extraction code.

```yaml
Layer1_DocumentStructure:
  Book:
    required: [book_id, title, author]
  Chapter:
    required: [chapter_id, chapter_title, chapter_number]
    optional: [subtitle]
  Section:
    required: [section_id, heading_text, heading_level, parent_chapter_id]
  Paragraph:
    required: [paragraph_id, text, page_number, parent_section_id]
    optional: [is_heading, heading_level]

Layer2_SemanticConcepts:
  Concept:
    required: [concept_id, canonical_name]
    optional: [variants, definition_paragraph_id]
  Relationship:
    required: [subject_id, relation_type, object_id, source_paragraph_id]
    optional: [confidence, verb]
```

---

## Phase 0: Source Analysis (DO THIS FIRST!)

**âš ï¸ LESSON LEARNED**: We skipped this and paid for it repeatedly.

### Step 0.1: EPUB Structure Exploration

| Step | Task | NLP Pipeline Name | Rationale | Output |
|------|------|-------------------|-----------|--------|
| 0.1.1 | Load single EPUB, examine raw HTML | **Source Format Analysis** | Understand what you're working with BEFORE building extraction | `epub_structure_report.md` |
| 0.1.2 | Document ALL HTML tag types used | **Markup Inventory** | EPUBs vary wildly; Tharpa uses `<p class="...">` not `<h1>` | Tag inventory list |
| 0.1.3 | Document ALL CSS classes for headings | **Style Pattern Discovery** | "Chapter-title-TOC-Level-1", "Section-Head", etc. | CSS class mapping |
| 0.1.4 | Manually count chapters/sections in one book | **Ground Truth Establishment** | Need baseline to validate against | Manual count document |
| 0.1.5 | Identify paragraph delimiter patterns | **Boundary Detection** | `<p>` tags? `\n\n`? Both? | Delimiter specification |

### ðŸš¦ VALIDATION GATE 0: Source Understanding

```python
def gate_0_source_understanding():
    """Must answer ALL these questions before Phase 1"""
    questions = [
        "What HTML tags are used for chapter titles?",      # Answer: <p class="Chapter-title-*">
        "What HTML tags are used for section headings?",    # Answer: <p class="Section-Head-*">
        "What HTML tags are used for body paragraphs?",     # Answer: <p class="Text-*">
        "How many chapters does Clear Light have?",         # Answer: ~12 real chapters
        "How many EPUB sections does it have?",             # Answer: 88 files
        "Are there nested heading levels?",                 # Answer: Yes, h1/h2/h3 equivalent
    ]
    # If you can't answer these, DO NOT proceed to Phase 1
```

**What we missed**: The EPUB uses `<p class="Chapter-title-TOC-Level-1">` for chapter titles, NOT `<h1>` tags. We looked for `<h1>` and found nothing.

---

## Phase 1: Data Preparation & Vector Database

### Step 1.1: EPUB Extraction (WITH Structure Preservation)

| Step | Task | NLP Pipeline Name | Rationale | Output |
|------|------|-------------------|-----------|--------|
| 1.1.1 | Extract text preserving `<p>` tag boundaries | **Structure-Preserving Extraction** | Honor sacred text boundaries; paragraphs are pedagogical units | `extracted_text/*.json` |
| 1.1.2 | **Extract chapter titles from CSS classes** | **Heading Extraction** | Look for `class="Chapter-title-*"`, NOT `<h1>` tags | Chapter title metadata |
| 1.1.3 | **Extract section headings from CSS classes** | **Section Hierarchy Extraction** | Look for `class="Section-Head-*"` patterns | Section heading metadata |
| 1.1.4 | **Assign heading levels based on CSS class** | **Hierarchy Level Assignment** | Map CSS classes to h1/h2/h3 equivalents | `heading_level` field |
| 1.1.5 | Preserve page number mappings | **Page Boundary Tracking** | Enable citation: "page 42" | `page_number` field |

**âš ï¸ LESSON LEARNED**: Our original extraction code:
```python
# WRONG - looked for tags that don't exist in this EPUB
heading = soup.find(['h1', 'h2', 'h3'])
if heading:
    chapter_title = heading.get_text().strip()
# Result: chapter_title = None for 87/88 chapters

# CORRECT - look for CSS classes used by Tharpa Publications
heading = soup.find('p', class_=lambda c: c and 'Chapter-title' in c)
if heading:
    chapter_title = heading.get_text().strip()
```

**âš ï¸ LESSON LEARNED**: Our original extraction destroyed paragraphs:
```python
# WRONG - destroyed all paragraph structure
text = re.sub(r'\s+', ' ', text)

# CORRECT - preserve paragraph boundaries
paragraphs = soup.find_all('p')
text = '\n\n'.join([p.get_text().strip() for p in paragraphs])
```

### ðŸš¦ VALIDATION GATE 1A: Extraction Quality

```python
def gate_1a_extraction_quality():
    """Run after EPUB extraction, BEFORE proceeding"""
    with open('extracted_text/Clear_Light_of_Bliss.json') as f:
        data = json.load(f)
    
    # Check 1: Do we have chapter titles?
    chapters_with_titles = sum(1 for ch in data['chapters'] if ch.get('chapter_title'))
    assert chapters_with_titles >= 10, f"Only {chapters_with_titles} chapters have titles!"
    
    # Check 2: Do we have realistic paragraph counts?
    total_paras = sum(len(ch.get('paragraphs', [])) for ch in data['chapters'])
    assert total_paras > 1000, f"Only {total_paras} paragraphs - structure likely destroyed"
    
    # Check 3: Show me a chapter title
    print("Sample chapter titles:")
    for ch in data['chapters'][:5]:
        print(f"  Ch {ch['chapter_index']}: {ch.get('chapter_title', 'NULL')}")
    
    # Check 4: Show me paragraph structure
    ch30 = data['chapters'][30]
    print(f"\nChapter 30 has {len(ch30['paragraphs'])} paragraphs")
    
    # HUMAN VERIFICATION: Do these look right?
```

### Step 1.2-1.5: Chunking and Embedding

| Step | Task | NLP Pipeline Name | Rationale | Output |
|------|------|-------------------|-----------|--------|
| 1.2.1 | Chunk text with 33% overlap | **Semantic Chunking with Overlap** | Ensure no information loss at boundaries | *(intermediate)* |
| 1.2.2 | **Respect paragraph boundaries when chunking** | **Boundary-Aware Chunking** | Never split mid-paragraph; sacred text integrity | Aligned chunks |
| 1.3.1 | Generate embeddings | **Dense Vector Encoding** | Convert text to 384-dimensional vectors | `embeddings/*.json` |
| 1.4.1 | Populate ChromaDB | **Vector Index Population** | Enable fast approximate nearest-neighbor retrieval | `vector_db/` |

### ðŸš¦ VALIDATION GATE 1B: Vector Database Quality

```python
def gate_1b_vector_quality():
    """Run after ChromaDB population"""
    # Query test
    results = collection.query(query_texts=["clear light meditation"], n_results=3)
    
    # Check: Results should be coherent paragraphs, not fragments
    for doc in results['documents'][0]:
        assert not doc.endswith('-'), "Chunk splits mid-word!"
        assert len(doc) > 100, "Chunk too short - may be fragment"
        print(f"Sample result: {doc[:200]}...")
```

---

## Phase 2: NLP Pipeline (NER + Relation Extraction)

### Step 2.1-2.3: Vocabulary Discovery and Curation

| Step | Task | NLP Pipeline Name | Rationale | Output |
|------|------|-------------------|-----------|--------|
| 2.1.1 | Extract candidate terms via POS patterns | **Terminology Extraction / Vocabulary Discovery** | Find domain-specific terms by analyzing corpus itself | `*_vocab_discovery.ipynb` |
| 2.2.1 | Filter by frequency + expert curation | **Domain Lexicon Curation** | Balance automation with human expertise | `03_cleaned_terms.json` |
| 2.3.1 | Finalize vocabulary with categories | **Lexicon Structuring** | Organize by grammatical function | `04_final_vocabulary.json` |

### ðŸš¦ VALIDATION GATE 2A: Vocabulary Quality

```python
def gate_2a_vocabulary_quality():
    """Run after vocabulary curation"""
    with open('04_final_vocabulary.json') as f:
        vocab = json.load(f)
    
    # Check: Counts match
    claimed = vocab['metadata']['total_terms']
    actual = sum(len(vocab['data'][cat]) for cat in vocab['data'])
    assert claimed == actual, f"Metadata says {claimed}, actual is {actual}"
    
    # Check: Key terms present
    all_terms = [t[0] for cat in vocab['data'].values() for t in cat]
    required_terms = ['clear light', 'emptiness', 'illusory body', 'central channel']
    for term in required_terms:
        assert term in all_terms, f"Missing critical term: {term}"
    
    print(f"âœ“ Vocabulary validated: {actual} terms")
```

### Step 2.4-2.7: NER Configuration and Relation Extraction

| Step | Task | NLP Pipeline Name | Rationale | Output |
|------|------|-------------------|-----------|--------|
| 2.4.1 | Generate EntityRuler patterns (3 case variants) | **Pattern-Based NER Configuration** | terms Ã— 3 variants = patterns; ensures recall | *(runtime)* |
| 2.5.1 | Add EntityRuler to spaCy pipeline | **Custom NER Integration** | `before="ner"` gives custom patterns priority | *(pipeline config)* |
| 2.6.1 | Develop relation extraction function | **Dependency-Based Relation Extraction** | Extract semantic triples via syntactic structure | `extract_*_relationships_v3()` |
| 2.7.1 | Validate accuracy on test paragraphs | **Extraction Quality Assessment** | Iterative refinement until target accuracy | `05_phase2_nlp_complete.json` |

### ðŸš¦ VALIDATION GATE 2B: NER + RE Quality

```python
def gate_2b_ner_re_quality():
    """Run after NER/RE development"""
    test_cases = [
        {
            'text': "Clear light is inseparable from emptiness.",
            'expected_entities': ['clear light', 'emptiness'],
            'expected_relation': 'inseparable from'
        },
        {
            'text': "The practice depends upon the teacher.",
            'expected_entities': ['practice', 'teacher'],
            'expected_relation': 'depends upon'
        }
    ]
    
    for case in test_cases:
        doc = nlp(case['text'])
        entities = [ent.text.lower() for ent in doc.ents]
        rels = extract_buddhist_relationships_v3(doc)
        
        for expected in case['expected_entities']:
            assert expected in entities, f"Missing entity: {expected}"
        
        rel_types = [r['relation'].lower() for r in rels]
        assert case['expected_relation'] in rel_types, f"Missing relation: {case['expected_relation']}"
    
    print("âœ“ NER + RE validated on test cases")
```

---

## Phase 2.5: Missing NLP Components (FROM YOUR OWN PDF!)

**âš ï¸ LESSON LEARNED**: The PDF specification included these, but we never implemented them.

### Step 2.5.1: Coreference Resolution

| Step | Task | NLP Pipeline Name | Rationale | Output |
|------|------|-------------------|-----------|--------|
| 2.5.1.1 | Install coreference model | **Coreference Setup** | NeuralCoref or AllenNLP | *(pipeline component)* |
| 2.5.1.2 | Resolve "this practice" â†’ specific technique | **Pronoun Resolution** | "It is inseparable..." â†’ what is "it"? | Resolved references |
| 2.5.1.3 | Maintain entity continuity across paragraphs | **Cross-Paragraph Linking** | Same concept mentioned in para 1 and para 5 | Entity chains |

**Why this matters**: Without coreference resolution:
- "This meditation leads to clear light" â†’ We don't know what "this meditation" refers to
- We lose cross-paragraph relationships

### Step 2.5.2: Concept Normalization

| Step | Task | NLP Pipeline Name | Rationale | Output |
|------|------|-------------------|-----------|--------|
| 2.5.2.1 | Create canonical form mapping | **Terminology Normalization** | "clear light" = "Clear Light" = "the clear light" | Canonical mapping dict |
| 2.5.2.2 | Map variants to canonical IDs | **Variant Resolution** | All forms â†’ single concept node in Neo4j | Normalized entities |
| 2.5.2.3 | Handle cross-tradition terms | **Cross-Tradition Mapping** | "sunyata" = "emptiness" = "Å›Å«nyatÄ" | Unified concept IDs |

**Why this matters**: Without normalization:
- Neo4j will have separate nodes for "Clear light", "clear light", "the clear light"
- Relationship queries will miss connections

### ðŸš¦ VALIDATION GATE 2C: Normalization Quality

```python
def gate_2c_normalization():
    """Run after implementing normalization"""
    test_variants = [
        ("Clear light", "clear_light"),
        ("clear light", "clear_light"),
        ("the clear light", "clear_light"),
        ("Clear Light", "clear_light"),
    ]
    
    for variant, expected_canonical in test_variants:
        result = normalize_concept(variant)
        assert result == expected_canonical, f"{variant} â†’ {result}, expected {expected_canonical}"
    
    print("âœ“ Normalization validated")
```

---

## Phase 3: Document Structure & Graph Database

### Step 3.1: Document Structure Parsing

| Step | Task | NLP Pipeline Name | Rationale | Output |
|------|------|-------------------|-----------|--------|
| 3.1.1 | Parse chapter boundaries from JSON | **Document Structure Parsing** | Identify physical hierarchy from formatting | *(intermediate)* |
| 3.1.2 | **Verify chapter titles are populated** | **Title Validation** | Must have titles, not null | Validated structure |
| 3.1.3 | **Extract section headings with levels** | **Heading Hierarchy Extraction** | h1 â†’ h2 â†’ h3 equivalent structure | Heading tree |
| 3.1.4 | Split on `\n\n` for paragraphs | **Paragraph Segmentation** | Recover teaching units | *(intermediate)* |
| 3.1.5 | Assign unique IDs and page mapping | **Provenance ID Assignment** | Enable citation tracking | *(intermediate)* |
| 3.1.6 | Build Layer 1 hierarchy | **Document Graph Construction** | Book â†’ Chapters â†’ Sections â†’ Paragraphs | `06_document_structure_layer1.json` |

### ðŸš¦ VALIDATION GATE 3A: Document Structure Quality

```python
def gate_3a_document_structure():
    """Run after document structure parsing - THIS IS WHERE WE FAILED"""
    with open('06_document_structure_layer1.json') as f:
        data = json.load(f)
    
    # Check 1: Chapter titles exist
    null_titles = sum(1 for ch in data['chapters'] if ch.get('chapter_title') is None)
    total_chapters = len(data['chapters'])
    assert null_titles < total_chapters * 0.1, f"{null_titles}/{total_chapters} chapters have null titles!"
    
    # Check 2: Realistic paragraph count
    assert data['total_paragraphs'] > 1000, f"Only {data['total_paragraphs']} paragraphs"
    
    # Check 3: Heading hierarchy exists
    has_headings = any(
        para.get('heading_level') is not None 
        for ch in data['chapters'] 
        for para in ch.get('paragraphs', [])
    )
    assert has_headings, "No heading levels detected!"
    
    # Check 4: Show me samples for human verification
    print("Sample chapter titles:")
    for ch in data['chapters'][10:15]:
        print(f"  Ch {ch['chapter_index']}: {ch.get('chapter_title', 'NULL')}")
    
    print("\nâœ“ Document structure validated")
```

**âš ï¸ THIS IS WHERE WE FAILED**: We never ran this validation. If we had, we would have seen:
```
Sample chapter titles:
  Ch 10: NULL
  Ch 11: NULL
  Ch 12: NULL
  Ch 13: NULL
  Ch 14: NULL

AssertionError: 87/88 chapters have null titles!
```

### Step 3.2: NLP Extraction with Provenance

| Step | Task | NLP Pipeline Name | Rationale | Output |
|------|------|-------------------|-----------|--------|
| 3.2.1 | Load vocabulary | **Domain Lexicon Loading** | Retrieve curated terms for NER | `04_final_vocabulary.json` |
| 3.2.2 | Regenerate EntityRuler patterns | **Pattern-Based NER Configuration** | Rebuild from portable vocabulary | *(runtime)* |
| 3.2.3 | Loop through paragraphs | **Corpus Iteration / Document Batching** | Process at natural unit boundaries | â€” |
| 3.2.4 | Run spaCy + extraction function | **NER + Dependency-Based Relation Extraction** | Identify entities, extract triples | â€” |
| 3.2.5 | **Normalize extracted concepts** | **Concept Normalization** | Canonical forms for graph nodes | â€” |
| 3.2.6 | Attach source metadata | **Provenance Annotation** | Link to paragraph_id, chapter, page | â€” |
| 3.2.7 | Save incrementally by chapter | **Checkpoint Serialization** | Fault tolerance; resume capability | â€” |
| 3.2.8 | Output semantic relationships | **Knowledge Base Export** | Structured format for graph import | `07_semantic_relationships.json` |

### ðŸš¦ VALIDATION GATE 3B: Extraction with Provenance

```python
def gate_3b_extraction_provenance():
    """Run after NLP extraction"""
    with open('07_semantic_relationships.json') as f:
        data = json.load(f)
    
    # Check 1: Relationships exist
    assert len(data['relationships']) > 500, f"Only {len(data['relationships'])} relationships"
    
    # Check 2: Provenance is complete
    sample = data['relationships'][0]
    assert 'source' in sample, "Missing source provenance!"
    assert 'paragraph_id' in sample['source'], "Missing paragraph_id!"
    assert 'page_number' in sample['source'], "Missing page_number!"
    
    # Check 3: Can we trace back to source text?
    para_id = sample['source']['paragraph_id']
    # Load document structure and verify paragraph exists
    with open('06_document_structure_layer1.json') as f:
        doc = json.load(f)
    
    found = False
    for ch in doc['chapters']:
        for para in ch.get('paragraphs', []):
            if para['paragraph_id'] == para_id:
                found = True
                print(f"âœ“ Traced relationship to source:")
                print(f"  {sample['subject']} --[{sample['relation']}]--> {sample['object']}")
                print(f"  Source: {para['text'][:100]}...")
                break
    
    assert found, f"Could not find source paragraph: {para_id}"
    print("\nâœ“ Provenance chain validated")
```

### Step 3.3: Neo4j Graph Database

| Step | Task | NLP Pipeline Name | Rationale | Output |
|------|------|-------------------|-----------|--------|
| 3.3.1 | Install Neo4j | **Graph Database Setup** | Desktop recommended for learning | â€” |
| 3.3.2 | Define schema + constraints | **Graph Schema Definition** | Node types, relationships, indexes | â€” |
| 3.3.3 | Load Layer 1 (structure) | **Structural Graph Population** | Book/Chapter/Section/Paragraph nodes | â€” |
| 3.3.4 | Load Layer 2 (semantics) | **Semantic Graph Population** | Concept nodes + relationship edges | â€” |
| 3.3.5 | Run validation query | **Cross-Layer Validation** | Confirm provenance linking works | â€” |

### ðŸš¦ VALIDATION GATE 3C: Neo4j End-to-End

```cypher
// This query must return results
MATCH (c1:Concept {name: "clear light"})-[r:RELATES_TO]->(c2:Concept {name: "emptiness"})
MATCH (para:Paragraph {paragraph_id: r.source_paragraph_id})
MATCH (ch:Chapter {chapter_index: para.chapter_index})
RETURN c1.name, r.relation, c2.name, ch.title, para.page_number, substring(para.text, 0, 200)
LIMIT 5
```

If this returns empty results, something is broken in the pipeline.

---

## Phase 3.4: Graph Refinement (Bridge Building & Cleanup)

**Purpose**: Address structural gaps and vocabulary issues discovered when we
explored the post-Phase-3.3 graph in the Neo4j Browser. Prerequisites for
Phase 4 hybrid retrieval.

**Origin**: After Phase 3.3 populated 3,533 nodes and 683 relationships, the
first thing we did was open the Browser and look. Three issues surfaced that
were invisible in the JSON files alone:

1. **The two sub-graphs are disconnected** (KI-003)
2. **Relationship vocabulary is noisy** (KI-002)
3. **Paragraph text has lost word-boundary spaces** (KI-001)

Items 1 and 2 are addressed here in Phase 3.4. Item 3 is deferred — see KI-001
in the Known Issues section.

### Step 3.4a: Build Concept→Paragraph Bridge

| Step | Task | NLP Pipeline Name | Rationale | Output |
|------|------|-------------------|-----------|--------|
| 3.4a.1 | Load `07_semantic_relationships.json` and extract (concept, citation) pairs from both `subject` and `object` of each record | **Provenance Pair Extraction** | The JSON already carries the paragraph evidence; both ends of a relationship are concept appearances | in-memory pairs |
| 3.4a.2 | Sample paragraph citations from JSON and verify Neo4j has matching nodes | **Cross-Source Format Validation** | Different citation format = silent failure during write | Validation flag |
| 3.4a.3 | Dry-run preview: edges that would be created; identify JSON-only or graph-only concepts | **Pre-Write Audit** | Surface mismatches before mutating the graph | Preview output |
| 3.4a.4 | Write `MENTIONS` edges via `MERGE` (idempotent, batched) | **Graph Bridge Construction** | Connect Concept layer to Paragraph layer | Updated Neo4j |
| 3.4a.5 | Re-run originally-failing query in Browser as visual confirmation | **Visual Validation** | "Show me" test makes the bridge tangible | Confirmed bridge |

**Deliverable**: `phase3_4a_build_concept_paragraph_bridge.ipynb`

### 🚦 VALIDATION GATE 3.4A: Bridge Built

```cypher
// Must return non-zero after 3.4a runs successfully
MATCH (c:Concept {canonical_form: 'clear_light'})-[:MENTIONS]->(p:Paragraph)
RETURN count(p) AS paragraph_count
// Expected: ~64 paragraphs (matches JSON pre-computation)
```

Per-concept paragraph counts in the graph must match per-concept paragraph
counts derived from `07_semantic_relationships.json` for the top concepts.

### Step 3.4b: Clean up noisy concept-concept relationships

| Step | Task | NLP Pipeline Name | Rationale | Output |
|------|------|-------------------|-----------|--------|
| 3.4b.1 | Inventory all relationship types in the graph with counts | **Relationship Vocabulary Audit** | Know what's there before deciding policy | Type:count table |
| 3.4b.2 | Classify each type as doctrinal / grammatical / structural | **Relationship Classification** | Policy applied per category, not globally | Classification map |
| 3.4b.3 | Relabel grammatical edges (`OF`, `AND`, `ON`, `PREPOSITION`-tagged, `CONJUNCTION`-tagged) to `CO_OCCURS_WITH` | **Edge Relabeling** | Preserve co-occurrence signal without falsely claiming doctrinal meaning | Updated Neo4j |
| 3.4b.4 | Verify doctrinal verbs unchanged; verify grammatical types gone | **Post-Refactor Validation** | Confirm scope of change matches intent | Verification query |

**Deliverable**: `phase3_4b_clean_relationship_vocabulary.ipynb`

**Policy decision** (committed for v2.1): **relabel, not delete**. Co-occurrence
is real information; the only error was calling it doctrinal. Queries that
need only doctrinal links can filter to specific types; queries that want
broad concept clustering can include `CO_OCCURS_WITH`.

### 🚦 VALIDATION GATE 3.4B: Vocabulary Cleaned

```cypher
// After 3.4b, this query should show:
// - Doctrinal types preserved (DEPENDS_UPON, ARISE_FROM, etc.)
// - A new CO_OCCURS_WITH type with the bulk of edges
// - No remaining OF, AND, ON between Concepts

MATCH (a:Concept)-[r]-(b:Concept)
RETURN type(r) AS rel_type, count(*) AS count
ORDER BY count DESC
```

### Step 3.4c: Text normalization — DEFERRED

See **KI-001** in the Known Issues section. Addressed during multi-book
extractor rebuild, not now. Test cases are captured in KI-001 so nothing is
lost.

---

## Phase 4: Dual-Mode RAG Integration

**Prerequisites**:
- Phase 3.4a complete (Concept→Paragraph bridge exists)
- Phase 3.4b complete (relationship vocabulary cleaned)

**Known limitations entering Phase 4**: KI-001 (text-spacing artifacts in
paragraph content) is unresolved. Quoted passages from the corpus will preserve
those surface errors. Track this as a known quality limit; do not let it block
Phase 4.

| Step | Task | NLP Pipeline Name | Rationale |
|------|------|-------------------|-----------|
| 4.1 | Academic query mode | **Hybrid Retrieval (Graph + Vector)** | Vector search for paragraphs, graph traversal for concept-relationship expansion |
| 4.2 | Guided meditation mode | **Sequence Retrieval with Timing** | OrganicMeditationTimer for natural pauses; visualization sequences sourced from Clear Light of Bliss |
| 4.3 | Scale to full corpus | **Corpus Expansion** | Same pipeline applied to other 25 books — this is the trigger for resolving KI-001 |
| 4.4 | Adapt for Tibetan | **Cross-Lingual Adaptation** | New EntityRuler patterns and concept normalization for Tibetan source texts |

---

## Summary: What Was Missing and When

| Phase | What We Did | What We Should Have Done | Consequence |
|-------|-------------|--------------------------|-------------|
| **0** | Skipped entirely | EPUB structure exploration, CSS class inventory | Looked for wrong HTML tags |
| **1.1** | `soup.find(['h1','h2','h3'])` | `soup.find('p', class_=lambda c: 'Chapter-title' in c)` | 87/88 null chapter titles |
| **1.1** | `re.sub(r'\s+', ' ', text)` | Preserve `\n\n` paragraph breaks | Destroyed paragraph structure |
| **1.x** | No validation gate | Gate 1A: verify titles and paragraph counts | Built on broken foundation |
| **2.x** | Skipped coreference | NeuralCoref for pronoun resolution | "This practice" â†’ unknown |
| **2.x** | Skipped normalization | Canonical form mapping | Duplicate nodes in Neo4j |
| **3.1** | No validation gate | Gate 3A: verify chapter titles populated | Proceeded with null titles |

---

## The "Show Me" Test

Before declaring ANY phase complete, you must be able to answer:

| Phase | Show Me Question |
|-------|------------------|
| 0 | "Show me the CSS classes used for chapter headings in this EPUB" |
| 1 | "Show me the title of Chapter 15" |
| 1 | "Show me 5 paragraphs from Chapter 30" |
| 2 | "Show me the canonical form for 'Clear Light'" |
| 2 | "Show me what 'this practice' resolves to in paragraph X" |
| 3 | "Show me a relationship and its source paragraph text" |
| 3 | "Show me all concepts related to 'emptiness' in Neo4j" |
| 3.4a | "Show me the paragraphs that mention `clear_light`" |
| 3.4b | "Show me the relationship-type vocabulary after cleanup" |
| 4.1 | "Show me a cited answer to 'What does Geshe-la teach about clear light?'" |
| 4.2 | "Show me a 10-minute clear-light visualization sequence with organic timing" |

If you can't show it, it's not done.

---

## File Naming Convention

```
00_* - Source analysis and exploration
01_* - Initial extraction/raw data
02_* - Cleaned/preprocessed data
03_* - Intermediate processing results
04_* - Final vocabulary/lexicon
05_* - NLP pipeline specification
06_* - Document structure (Layer 1)
07_* - Semantic relationships (Layer 2)
08_* - Graph database exports
09_* - Integration/query results
```

---

## Complete Checklist Before Neo4j

- [ ] Phase 0 complete: Know exactly what HTML/CSS patterns EPUB uses
- [ ] Chapter titles extracted (not null)
- [ ] Section headings extracted with levels
- [ ] Paragraph boundaries preserved
- [ ] Vocabulary validated (term counts match)
- [ ] Coreference resolution implemented
- [ ] Concept normalization implemented
- [ ] Phase 3.4a complete: Concept→Paragraph bridge built and verified
- [ ] Phase 3.4b complete: Grammatical relationships relabeled as `CO_OCCURS_WITH`
- [ ] KI-001 noted, test cases captured, scheduled for extractor rebuild
- [ ] All Known Issues have current status and trigger conditions
- [ ] All validation gates passed
- [ ] "Show Me" tests answered for each phase

---

## Philosophical Foundation

> "Like Tibetan lotsÄwas (translators) 1000 years ago who knew every translation was transformation, we acknowledge all processing transforms text. The question is: Which transformation better honors the source?"

1. **All processing transforms** â€” The question is *how* we transform
2. **Skillful means matter** â€” Technical choices reflect spiritual values
3. **Structure preserves meaning** â€” Paragraph boundaries are pedagogical boundaries
4. **Precision serves devotion** â€” Accurate citations honor the teaching lineage
5. **Validate before proceeding** â€” Broken foundations corrupt everything built upon them

---

*Generated: January 2026*
*Project: Buddhist RAG System*
*Version: 2.1 - Phase 3.4 Refinement Edition*
*Current Status: Phase 3.3 complete; Phase 3.4a in progress; Phase 4 pending Phase 3.4 completion*
*Last revised: June 17, 2026*
