# Buddhist RAG System: Complete NLP Pipeline Reference (v2.2)

## Lessons Learned Edition

**Key Insight**: This document now includes **validation gates** and **exploratory analysis steps** that were missing in v1, leading to repeated rework. Each gate must pass before proceeding.

---

## Revision Log

- **v2.2 (June 24, 2026)** — **Phase 3.4 COMPLETE.** Marked KI-002 and KI-003 as RESOLVED with notebook references. Added KI-004 (self-loops, minor finding from 3.4b). Updated Complete Checklist to reflect Phase 3.4 done. System now has 5,077 total relationships including 912 Concept→Paragraph `MENTIONS` edges and a cleanly separated `CO_OCCURS_WITH` (438) vs. doctrinal verbs (~230) split.
- **v2.1 (June 17, 2026)** — Added "Known Issues & Deferred Work" section capturing KI-001 (lost word-boundary spaces), KI-002 (noisy relationship vocabulary), KI-003 (disconnected sub-graphs). Added Phase 3.4 "Graph Refinement" between Phase 3 and Phase 4. Updated Phase 4 prerequisites and Complete Checklist accordingly.
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

**⚠️ Standing inventory of corpus-quality issues discovered during exploration. Items here are tracked through their lifecycle — DEFERRED, SCHEDULED, IN PROGRESS, or RESOLVED.**

### How this section works

Every entry has:
- **ID** (KI-NNN) for cross-referencing across the doc and chat sessions
- **Status** — current state; updated as items move through the pipeline
- **Discovered** — where/when we first noticed the issue
- **Examples / Test cases** — concrete data points so we know what "fixed" looks like
- **Trigger condition / Resolution** — what causes us to address this, or what notebook resolved it
- **Rationale** — why deferred or how it was resolved

When an item is resolved, its status changes to `RESOLVED in Phase X.Y` with a reference to the resolving notebook. It is NOT removed from this section — the inventory is historical as well as forward-looking.

---

### KI-001: Lost word-boundary spaces in paragraph text

**Status**: DEFERRED — to multi-book extractor rebuild

**Discovered**: June 17, 2026 exploration session. Reading paragraphs `CLB.6.p1` and `CLB.6.p2` in the Neo4j Browser revealed systematic missing spaces at word boundaries.

**Root cause**: EPUB-to-text extraction (Phase 1.1) stripped HTML tags around italicized or styled inline text without preserving the implicit whitespace those tags carried. Words flanking formatting markup got concatenated.

**Examples** (test cases for the eventual fix):

| Observed | Should be |
|---|---|
| `Tsongkhapa'sLamp` | `Tsongkhapa's Lamp` |
| `commentary to theSix Yogas` | `commentary to the Six Yogas` |
| `Mahamudra,TheMain Path` | `Mahamudra, The Main Path` |
| `ofMahamudra` | `of Mahamudra` |
| `asJoyful Path` | `as Joyful Path` |
| `variouspreliminaries` | `various preliminaries` |
| `a qualifiedVajrayana` | `a qualified Vajrayana` |
| `Tantraempowerment` | `Tantra empowerment` |
| `ourvows andcommitments` | `our vows and commitments` |
| `attainedBuddhahood` | `attained Buddhahood` |
| `arefive stages` | `are five stages` |
| `theillusory body` | `the illusory body` |
| `aDeity` | `a Deity` |
| `gseven pre-eminent` | `g seven pre-eminent` |

**Pattern**: lowercase-letter immediately followed by uppercase-letter (`sL`, `oM`, `tT`), or punctuation immediately followed by letter (`,T`, `.A`), are the most common indicators. **NOT always** — proper compounds (`MacBook`, `iPhone`) match the same pattern, so any regex needs a denylist.

**Trigger condition**: Address when extending extraction to a second book. The cheapest fix is upstream in the extractor itself.

**What this means for Phase 4 results**: Quoted passages in Claude's responses will preserve these surface errors. Acceptable as a temporary quality limitation. Disclose as a data caveat in Phase 4 output until resolved.

**Rationale for deferral**: Single-book regex pass would need to run again for every new book. Fixing in the extractor is a one-time cost. Phase 4 retrieval embeddings are robust to small surface noise. Test cases captured above so nothing is lost.

---

### KI-002: Noisy doctrinal-relationship vocabulary

**Status**: ✅ **RESOLVED in Phase 3.4b** — `phase3_4b_clean_relationship_vocabulary.ipynb`

**Discovered**: June 17, 2026 exploration session. Relationship-type profile around `clear_light` returned: `OF: 71`, `DEPENDS_UPON: 10`, `AND: 4`, etc. The most common relationship type for the most-mentioned tantric concept was the English preposition `OF`.

**Root cause**: Phase 2/3 relation extractor used grammatical surface markers (prepositions, conjunctions) as relationship types instead of either filtering them or aggregating them under a non-doctrinal label. `07_semantic_relationships.json` metadata classified 311 of 683 relationships as `PREPOSITION` and 125 as `CONJUNCTION`, but the surface label was preserved when ingested into Neo4j.

**Resolution** (June 24, 2026): Phase 3.4b relabeled 438 grammatical edges as `CO_OCCURS_WITH` and unified 15 tantric-instruction edges as `TANTRIC_INSTRUCTION`. Doctrinal verbs (`DEPENDS_UPON`, `ARISE_FROM`, `MIXING_WITH`, `DISSOLVE_WITHIN`, `KNOWN_AS`, `MEDITATING_ON`, `FREE_FROM`, `EMPTY_OF`, etc.) preserved unchanged. Original surface form preserved on every relabeled edge as the `original_type` property.

**Verification**: All doctrinal type counts match before/after exactly (16/16 ✓). `OF`/`AND`/`ON`/`WITH`/`OR` all reduced to 0 remaining. 683 total Concept↔Concept edges → still 683 (no loss).

---

### KI-003: Concept layer disconnected from Paragraph layer

**Status**: ✅ **RESOLVED in Phase 3.4a** — `phase3_4a_build_concept_paragraph_bridge.ipynb`

**Discovered**: June 17, 2026 exploration session. The query `MATCH (c:Concept {canonical_form: 'clear_light'})-[]-(p:Paragraph)` returned zero results — no edge type existed between Concept and Paragraph nodes after Phase 3.3.

**Root cause**: Phase 3.3 created two functional sub-graphs (Book→Chapter→Paragraph structural tree, Concept↔Concept doctrinal web) without building the bridge edges connecting them. The data for the bridge existed in `07_semantic_relationships.json` (every relationship's `source.citation` field), but it was used to set `mention_count` properties on Concept nodes instead of producing edges.

**Resolution** (June 24, 2026): Phase 3.4a built **912 `MENTIONS` edges** from each Concept to every Paragraph where it appears, using the `source.citation` field as the join key. Idempotent via `MERGE`. Per-concept paragraph counts in the graph match per-concept paragraph counts derived from JSON exactly (e.g., `clear_light=64/64`, `mind=132/132`, `emptiness=58/58`).

**What this unlocked**: Multi-hop graph traversal across the corpus. The query `(c1:Concept)-[:MENTIONS]->(p:Paragraph)<-[:MENTIONS]-(c2:Concept)` now returns paragraphs where two concepts co-appear — the doctrinal-intersection retrieval pattern that vector search alone cannot do. Demonstrated with `clear_light + illusory_body` returning the 11 paragraphs that form the doctrinal core of Clear Light of Bliss.

---

### KI-004: Self-loop relationships on canonical concepts

**Status**: NOTED — minor; no current action

**Discovered**: June 24, 2026 during Phase 3.4b verification. Several doctrinal claims surfaced with the same concept on both ends, e.g.:
- `clear_light --[KNOWN_AS]-- clear_light`
- `clear_light --[ARISE_FROM]-- clear_light`
- `clear_light --[ATTAINED_IN]-- clear_light`

**Likely cause**: Two distinct surface forms (e.g., "the clear light of death" and "the all-empty clear light") both canonicalize to `clear_light`, producing tautological-looking edges. The pipeline lost the qualifier that would have distinguished them.

**Status rationale**: Cosmetic at retrieval time — a query that returns `clear_light DEPENDS_UPON clear_light` is meaningless but not actively harmful. Address when concept normalization gets a more discriminating canonical-form policy (likely during multi-book extraction when more variants surface).

**Trigger condition**: When working on Phase 4 retrieval, if self-loops surface in user-facing output, add a Cypher filter (`WHERE a <> b`) to the query layer as a temporary measure.

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

**Status**: ✅ **COMPLETE** (June 24, 2026)

**Purpose**: Address structural gaps and vocabulary issues discovered when exploring the post-Phase-3.3 graph in the Neo4j Browser. Prerequisites for Phase 4 hybrid retrieval.

**Origin**: After Phase 3.3 populated 3,533 nodes and 683 relationships, the first action was to open the Browser and look. Three issues surfaced that were invisible in the JSON files alone:

1. **The two sub-graphs are disconnected** (KI-003)
2. **Relationship vocabulary is noisy** (KI-002)
3. **Paragraph text has lost word-boundary spaces** (KI-001)

Items 1 and 2 were addressed in Phase 3.4. Item 3 is deferred — see KI-001.

### Step 3.4a: Build Concept→Paragraph Bridge

**Deliverable**: `phase3_4a_build_concept_paragraph_bridge.ipynb`  
**Result**: 912 `MENTIONS` edges created. KI-003 resolved.

| Step | Task | NLP Pipeline Name | Rationale | Output |
|------|------|-------------------|-----------|--------|
| 3.4a.1 | Load `07_semantic_relationships.json` and extract (concept, citation) pairs from both `subject` and `object` of each record | **Provenance Pair Extraction** | The JSON already carries paragraph evidence; both ends of a relationship are concept appearances | in-memory pairs |
| 3.4a.2 | Sample paragraph citations from JSON and verify Neo4j has matching nodes | **Cross-Source Format Validation** | Different citation format = silent failure during write | Validation flag |
| 3.4a.3 | Dry-run preview: edges that would be created; identify JSON-only or graph-only concepts | **Pre-Write Audit** | Surface mismatches before mutating the graph | Preview output |
| 3.4a.4 | Write `MENTIONS` edges via `MERGE` (idempotent, batched) | **Graph Bridge Construction** | Connect Concept layer to Paragraph layer | Updated Neo4j |
| 3.4a.5 | Re-run originally-failing query in Browser as visual confirmation | **Visual Validation** | "Show me" test makes the bridge tangible | Confirmed bridge |

### 🚦 VALIDATION GATE 3.4A: Bridge Built ✓ PASSED

```cypher
MATCH (c:Concept {canonical_form: 'clear_light'})-[:MENTIONS]->(p:Paragraph)
RETURN count(p) AS paragraph_count
// Returned: 64 (matches JSON pre-computation exactly)
```

### Step 3.4b: Clean up noisy concept-concept relationships

**Deliverable**: `phase3_4b_clean_relationship_vocabulary.ipynb`  
**Result**: 453 edges relabeled (438 → `CO_OCCURS_WITH`, 15 → `TANTRIC_INSTRUCTION`). All doctrinal types preserved. KI-002 resolved.

| Step | Task | NLP Pipeline Name | Rationale | Output |
|------|------|-------------------|-----------|--------|
| 3.4b.1 | Inventory all relationship types between Concepts with counts | **Relationship Vocabulary Audit** | Know what's there before deciding policy | 36 distinct types found |
| 3.4b.2 | Use JSON `relation_type` classification as ground truth | **Classification Lookup** | Pipeline already classified; reuse not re-derive | Per-edge classification |
| 3.4b.3 | Relabel grammatical edges (PREPOSITION + CONJUNCTION → `CO_OCCURS_WITH`; TANTRIC_INSTRUCTION → unified) | **Edge Relabeling** | Preserve co-occurrence signal without falsely claiming doctrinal meaning | Updated Neo4j |
| 3.4b.4 | Verify doctrinal verbs unchanged; verify grammatical types removed | **Post-Refactor Validation** | Confirm scope of change matches intent | Verification query |

**Policy decision** (committed): **relabel, not delete**. Co-occurrence is real information; the only error was calling it doctrinal. Original surface forms preserved on every relabeled edge via `original_type` property — fully reversible if needed.

### 🚦 VALIDATION GATE 3.4B: Vocabulary Cleaned ✓ PASSED

```cypher
MATCH (a:Concept)-[r]-(b:Concept)
RETURN type(r) AS rel_type, count(*) AS count
ORDER BY count DESC
// After 3.4b:
// CO_OCCURS_WITH: 438
// DEPENDS_UPON: 27, MIXING_WITH: 26, DISSOLVE_WITHIN: 24, KNOWN_AS: 23, ...
// TANTRIC_INSTRUCTION: 15
// No remaining OF / AND / ON / WITH / OR
```

### Step 3.4c: Text normalization — DEFERRED

See **KI-001** in Known Issues. Addressed during multi-book extractor rebuild, not now. Test cases captured in KI-001.

### Phase 3.4 Final State

| Layer | Count |
|---|---|
| Total nodes | 3,533 |
| Total relationships | 5,077 |
| `HAS_CHAPTER` (Book→Chapter) | 33 |
| `HAS_PARAGRAPH` (Chapter→Paragraph) | 3,449 |
| `MENTIONS` (Concept→Paragraph) | **912** (NEW from 3.4a) |
| `CO_OCCURS_WITH` (Concept↔Concept) | **438** (relabeled in 3.4b) |
| `TANTRIC_INSTRUCTION` (Concept↔Concept) | **15** (unified in 3.4b) |
| Doctrinal verb edges (preserved) | 230 across 22 types |

The graph is now ready for Phase 4 hybrid retrieval.

---

## Phase 4: Dual-Mode RAG Integration

**Prerequisites**:
- ✅ Phase 3.4a complete (Concept→Paragraph bridge exists)
- ✅ Phase 3.4b complete (relationship vocabulary cleaned)

**Known limitations entering Phase 4**: KI-001 (text-spacing artifacts) is unresolved. Quoted passages from the corpus will preserve those surface errors. Track this as a known quality limit until extractor rebuild. KI-004 (self-loops) is minor and can be filtered in queries with `WHERE a <> b` if user-facing.

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
| 3.4a | "Show me the paragraphs that mention `clear_light`" ✅ |
| 3.4a | "Show me paragraphs that mention BOTH clear_light AND illusory_body" ✅ |
| 3.4b | "Show me only doctrinal claims about clear_light (no grammatical noise)" ✅ |
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

## Complete Checklist

- [x] Phase 0 complete: Know exactly what HTML/CSS patterns EPUB uses
- [x] Chapter titles extracted (not null)
- [x] Section headings extracted with levels
- [x] Paragraph boundaries preserved
- [x] Vocabulary validated (term counts match)
- [x] Concept normalization implemented
- [x] Neo4j populated (Phase 3.3)
- [x] **Phase 3.4a complete: Concept→Paragraph bridge built and verified** (June 24, 2026)
- [x] **Phase 3.4b complete: Grammatical relationships relabeled as CO_OCCURS_WITH** (June 24, 2026)
- [x] KI-002 RESOLVED via 3.4b
- [x] KI-003 RESOLVED via 3.4a
- [ ] KI-001 (text spacing) — DEFERRED to multi-book extractor rebuild
- [ ] KI-004 (self-loops) — NOTED, no current action
- [ ] Phase 4.1: Academic query mode (Hybrid Retrieval)
- [ ] Phase 4.2: Guided meditation mode (OrganicMeditationTimer)
- [ ] Phase 4.3: Scale to full corpus (also triggers KI-001 resolution)
- [ ] Phase 4.4: Tibetan adaptation
- [x] All validation gates passed (through Phase 3.4)
- [x] "Show Me" tests answered for each phase (through Phase 3.4)

---

## Philosophical Foundation

> "Like Tibetan lotsÄwas (translators) 1000 years ago who knew every translation was transformation, we acknowledge all processing transforms text. The question is: Which transformation better honors the source?"

1. **All processing transforms** â€” The question is *how* we transform
2. **Skillful means matter** â€” Technical choices reflect spiritual values
3. **Structure preserves meaning** â€” Paragraph boundaries are pedagogical boundaries
4. **Precision serves devotion** â€” Accurate citations honor the teaching lineage
5. **Validate before proceeding** â€” Broken foundations corrupt everything built upon them

---

*Originally generated: January 2026 (v2.0)*
*Latest revision: June 24, 2026 (v2.2)*
*Project: Buddhist RAG System (Geshe_La_Rag)*
*Current Status: Phase 3.4 complete — graph ready for Phase 4 hybrid retrieval.*
