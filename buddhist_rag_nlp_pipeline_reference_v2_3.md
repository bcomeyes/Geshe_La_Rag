# Buddhist RAG System: Complete NLP Pipeline Reference (v2.3)

## Lessons Learned Edition

**Key Insight**: This document now includes **validation gates** and **exploratory analysis steps** that were missing in v1, leading to repeated rework. Each gate must pass before proceeding.

---

## Philosophical Foundation

_This section was added in v2.3 and is the new center of gravity for the project. All engineering decisions below are tested against the principles articulated here. The five condensed touchstones below — preserved from v2.0 — capture the essence; the prose that follows elaborates and operationalizes them._

### Core Touchstones (preserved from v2.0)

> "Like Tibetan lotsāwas (translators) 1000 years ago who knew every translation was transformation, we acknowledge all processing transforms text. The question is: Which transformation better honors the source?"

1. **All processing transforms** — The question is *how* we transform
2. **Skillful means matter** — Technical choices reflect spiritual values
3. **Structure preserves meaning** — Paragraph boundaries are pedagogical boundaries
4. **Precision serves devotion** — Accurate citations honor the teaching lineage
5. **Validate before proceeding** — Broken foundations corrupt everything built upon them

### What this project actually is

The Geshe_La_Rag system is a **proof of concept** — not for itself, but for a larger arc of work the author is undertaking. That larger arc includes:

- OCR with grid search on Tibetan pecha
- Masked-transformer models for restoration of damaged and poor-quality folios
- An overarching LLM or SLM intended to help practitioners on the path

This RAG project sits at the architectural beginning of that arc. The patterns established here — how a corpus is extracted, how a vocabulary evolves, how retrieval orchestrates across stores, how responses honor their sources — will inform every subsequent piece of work. The Geshe_La_Rag is therefore not just a project; it is the place where the author practices building the kind of system that the larger work will require.

Clear Light of Bliss is the first text in this scope. The other 25 books by Geshe Kelsang Gyatso are the immediate next horizon. Beyond that, the wider Buddhist canon — through translation, through OCR of source pecha, through scholarly attribution — is the eventual scope. What is built in Phase 3.5 onward should accommodate this growth not by anticipating every requirement, but by refusing to assume "only one book, only one author, only one source."

### Why care matters here in a way that goes beyond engineering

The Buddhist canon is the longest continuously-curated body of liberatory literature in human history. The Kangyur and Tengyur exist because for over a millennium, generations of practitioners, scholars, and translators treated the texts as worth carrying forward exactly. The lineage is partly the texts and partly the **care** with which they were carried.

A digital system that surfaces these teachings to practitioners is a continuation of that carrying-forward, in a new medium. The infrastructure choices made here will be inherited by whatever this becomes. Future practitioners who use this system — or its descendants — will receive their teachings through architecture we built. That is a real responsibility, and it deserves a real standard.

There is also a more specific concern that Geshe-la himself writes about: the consequences of transmitting incorrect teachings. A teacher who passes on confused teachings — even with good intentions — bears karmic weight for the confusion their students experience. An LLM is not a teacher in any traditional sense. But the system that stands between the teacher's words and the practitioner's eye carries some weight that the teacher would otherwise carry directly. Errors of attribution, fused doctrinal claims that shouldn't be fused, misrenderings of cited text — these are not merely engineering bugs. They are places where the integrity of the lineage depends on the integrity of how the system was built.

This is not a reason for paralysis. It is a reason for **standard**. The teachings have always been carried forward carefully. Asking the same of the digital lineage is not a high bar; it is the bar.

### The guiding principle for engineering decisions

**Build Clear Light of Bliss as if it's the first of 26, not the only one. Build it as if it's the first text in a much wider canon-handling system, not the only one.**

This principle resolves a recurring tension. The temptation under time pressure is to make local optimizations: hardcode a single-book assumption, patch a downstream artifact, skip a validation step, defer a structural concern. Each individual shortcut is small. Their accumulation is not.

The principle inverts the default. When a decision would embed an assumption that book #2 (or canon-source #2) would later force us to undo, the principle asks: what is the small additional investment that removes the assumption? If that investment is cheap, we make it. If it is expensive, we name it explicitly as a deferred structural commitment and add it to the Known Issues backlog with a trigger condition. We do not silently embed assumptions.

This principle does not mean over-engineering. We do not build for every imaginable future. We build for the **known** scope (Geshe-la's 26 books, eventually the wider canon, eventually multi-source attribution) and we name the rest as out-of-scope. The discipline is to distinguish honestly between "premature abstraction" (avoid) and "foundational decision that will be expensive to retrofit" (address now).

### What honoring the material looks like in practice

These are not slogans; they are operational standards that this document tracks across the pipeline:

1. **The text as the teacher wrote it is sacred.** We do not restructure paragraphs, do not summarize source text, do not let the LLM rewrite Geshe-la's words. Where extraction has introduced errors (KI-001), we fix the extractor; we do not paper over the output.
2. **Citations are non-negotiable.** Every response that draws on the corpus carries a citation back to the source. Every assertion that goes beyond the citations is marked as inference, not teaching.
3. **No impersonation.** Claude as the response generator refers to Geshe-la in third person. We do not let the system speak as him. The teacher is the source; the system is the transmitter.
4. **Honest acknowledgment of limits.** When the corpus does not directly answer a question, the system says so. We do not confabulate. We do not let semantic similarity stand in for actual teaching content.
5. **Validation before propagation.** No write to a database happens without an explicit validation gate. No retrieval is added without a "show me" test that demonstrates it works against a real query. No architectural commitment is added without being named in this document.
6. **Errors found are errors documented.** Every issue discovered during exploration goes into Known Issues with a stable ID, a status, and a resolution path. Nothing is forgotten in chat history.

### A note on the karmic framing

The author of this project carries the karmic frame seriously, and this document acknowledges it without trying to formalize it beyond what is appropriate. The engineering discipline above is the operational expression of that frame. The reverse is also true: the discipline holds even without the karmic frame, because it is what serious engineering looks like for serious work. We do not need to overclaim the spiritual stakes to motivate doing the work carefully. We do it carefully because it deserves that, regardless of whether one chooses to read karmic weight into the engineering or not.

---

## Architectural Commitments (S1–S6)

_These are the operational consequences of the Philosophical Foundation. Each is structural — a decision that, once embedded, shapes everything that follows. Each is testable against the principle: "does this assume one book/one source in a way that would be expensive to undo?"_

### S1: Single source of truth for paragraph text

One extraction pipeline produces the canonical paragraph corpus. Every downstream artifact — Neo4j, ChromaDB, any future store — consumes that canonical output. There is no re-extraction in two places.

**Why it matters at scale**: When the corpus grows to 26 books and eventually to the wider canon, parallel extraction pipelines become parallel maintenance burdens and parallel sources of silent drift. One canonical extraction with multiple consumers is the only structure that scales.

**Current status**: In v2.2 the system has two independent extraction paths (Phase 1 → JSON → Neo4j, and a separate path that re-extracts from EPUB for ChromaDB). This is documented as KI-005 and resolved in Phase 3.5.

### S2: Pipeline as a parameterizable function, not a hardcoded notebook

The end-to-end pipeline (extraction → normalization → graph population → embedding) is callable as a function: `process_book(epub_path, book_config) -> ProcessedBook`. Notebooks are demos and learning artifacts. The underlying logic lives in an importable Python package (`geshe_rag/`).

**Why it matters at scale**: Running 26 hand-orchestrated notebooks is not feasible. The pipeline must accept parameters and produce a known artifact structure per book. Adding a 27th source becomes calling the function with a new config.

### S3: Per-source configuration as data, not code

Each corpus source has a configuration file (TOML or JSON) that captures its quirks: CSS class mappings, heading conventions, structural-role rules, front-matter patterns, any per-source extraction parameters. The extractor reads the config; it does not embed any single source's assumptions in the code.

**Why it matters at scale**: Tharpa's other books may diverge in styling from Clear Light of Bliss. Tibetan pecha will diverge dramatically. Translated sources will diverge differently again. Per-source config means "add a new source" is a configuration task, not a code change.

### S4: The canonical vocabulary is versioned and evolves through delta review

The concept normalization map (`04b_normalization_map.json` in v2.2) becomes the v1 canonical vocabulary. When a new source surfaces new concepts, the system produces a **vocabulary delta**: which concepts are new, which appear to be variants of existing concepts (proposed merges), which require human review. The canonical vocabulary advances through explicit versions; every processed source is tagged with the vocabulary version it was processed against.

**Why it matters at scale**: A vocabulary built solely on Clear Light of Bliss is biased toward its subject matter (tantric completion stage practice). A vocabulary that must grow across 26 books, then translation, then the wider canon, needs a disciplined growth model. Without one, every new source either contaminates the canonical concepts or sits in isolation from them.

### S5: Schema accommodates multi-source queries from the start

Even with one book in the graph, retrieval queries are written with explicit source filters (`MATCH (b:Book {code: 'CLB'})-[*]->(p:Paragraph)...`) rather than implicit single-source assumptions. Retrieval functions take an optional `sources=['CLB', ...]` parameter that defaults to "all sources currently in graph." For now that is just CLB; the parameter exists.

**Why it matters at scale**: A retrieval layer built around "there is one book, so we don't need to scope" is a layer that must be rewritten the moment a second book arrives. A retrieval layer that scopes from day one is a layer that grows naturally as sources are added.

### S6: Author and source attribution as first-class metadata

Every paragraph carries `author` and `source` metadata, even though in v2.2 every paragraph is Geshe Kelsang Gyatso writing in *Clear Light of Bliss*. When Tibetan source texts get added (Phase 4.4 and beyond), and when the wider canon brings in translators, commentators, and multiple lineage masters, the attribution machinery is already in place.

**Why it matters at scale**: A response prompt that assumes "the teacher is always Geshe-la" embeds an assumption that the wider canon would force us to undo. A response prompt that reads each paragraph's `author` field and attributes correctly is one that grows naturally into multi-author corpora.

### How these commitments are used

For any engineering decision going forward, the question is: *does this decision uphold or violate S1–S6?* If it violates any of them, either the decision needs to change or the violation needs to be named explicitly as a known structural debt with a resolution path in this document.

The commitments are not aspirations. They are tests.

---

## Revision Log

- **v2.3 (June 27, 2026)** — Added "Philosophical Foundation" section establishing this project as a proof-of-concept for a larger digital lineage that will eventually include OCR of Tibetan pecha, masked-transformer restoration of damaged folios, and an SLM for practice guidance. Articulated six structural commitments (S1-S6) that all engineering decisions are tested against. Restructured roadmap: Phase 3.5 (foundational rework) inserted before Phase 4, with the principle that Clear Light of Bliss is built "as if it's the first of 26, not the only one." Added KI-005 (two independent EPUB-to-text pipelines, to be resolved by Phase 3.5). Replaced KI-001's status from DEFERRED to SCHEDULED for Phase 3.5 (resolved via upstream extractor fix rather than downstream patch).
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

**Status**: SCHEDULED for resolution in Phase 3.5 (upstream fix via S1+S2)

**Resolution approach (revised in v2.3)**: Rather than a downstream regex patch on the current corpus, the fix is applied at the EPUB extraction step itself — using BeautifulSoup's `get_text(separator=' ')` parameter to insert word boundaries at every styled-text transition. This change happens once in the canonical extractor (per S1) and propagates naturally to all downstream artifacts. When the corpus expands to other books, no per-book regex patch is needed — the extractor produces clean text from the start.

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

### KI-005: Two independent EPUB-to-text extraction pipelines

**Status**: SCHEDULED for resolution in Phase 3.5 (via S1)

**Discovered**: June 27, 2026, during reconnaissance for the Phase 3.5 spacing-fix work. A grep for `get_text` across the project revealed that the pipeline that produces Neo4j paragraphs (`phase1_reextract_structure.ipynb`) and the pipeline that produces ChromaDB chunks (`enhanced_embedding_extraction_geshe_la.ipynb`) extract text from EPUBs **independently**, with separate code paths.

**Root cause**: The two pipelines were developed at different times for different purposes and were never unified. As a result, the same EPUB is processed twice through similar-but-not-identical extraction logic.

**Why this is a structural problem, not just a bug**: With two pipelines, any change to extraction logic (the spacing fix, future per-book configs, future extraction improvements) has to be applied and verified in two places. The two stores can silently drift apart. When the corpus grows to 26 books, the maintenance burden doubles per book. This violates S1 (single source of truth for paragraph text).

**Resolution path**: Phase 3.5 implements S1 by restructuring the embedding pipeline to consume the canonical Phase 1 JSON output rather than re-extracting from EPUBs. This is structural, not a workaround. The spacing fix (KI-001) is applied at the single canonical extraction point and propagates naturally to both stores.

**Verification at resolution time**: SRENE cross-store consistency check — for a random sample of paragraphs, the text in Neo4j must exactly match the text in the corresponding ChromaDB chunks.

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

## Phase 3.5: Structural Foundation Rework

**Status**: PLANNED (to be completed before Phase 4 work resumes)

### Purpose

Implement the six architectural commitments (S1–S6) against the existing Clear Light of Bliss corpus, so that Phase 4 is built on a foundation that scales logically (not necessarily easily) to the other 25 books and beyond.

### What it accomplishes

- **S1 implemented**: one canonical extraction pipeline; ChromaDB and Neo4j both consume its output
- **S2 implemented**: the pipeline is callable as `geshe_rag.process_book(epub_path, config)`; notebooks become learning/demo artifacts
- **S3 implemented**: `configs/clear_light_of_bliss.toml` captures this book's extraction parameters; the extractor reads from config rather than assuming
- **S4 staged**: vocabulary versioned as v1; delta-review machinery scaffolded for future sources
- **S5 implemented**: retrieval queries and functions accept source scope as parameter from day one
- **S6 implemented**: every paragraph carries `author` and `source` metadata explicitly

- **KI-001 resolved**: spacing fix applied at the canonical extraction point via `get_text(separator=' ')`
- **KI-005 resolved**: the two-pipeline problem dissolves because there is now one pipeline

### Sub-phase sequence

| Sub-phase | Work | Validation gate |
|---|---|---|
| 3.5.1 | Create `geshe_rag/` Python package skeleton with `extractor/`, `normalizer/`, `populator/`, `embedder/`, `retrieval/`, `generation/` modules | Package imports cleanly in a notebook |
| 3.5.2 | Move Phase 1 extraction logic into `geshe_rag.extractor.extract_book(epub_path, config)` with the spacing fix applied | Test cases from KI-001 pass against extracted output |
| 3.5.3 | Move Phase 2 normalization into `geshe_rag.normalizer.normalize(extracted_data, vocab_version)` | Same canonical-form output as Phase 2.5 produced |
| 3.5.4 | Move Phase 3 population into `geshe_rag.populator.populate_neo4j(normalized_data)`, with explicit book attribution and source scoping | Node and relationship counts match Phase 3.3 outputs |
| 3.5.5 | Refactor ChromaDB embedding to consume canonical Phase 1 JSON output (no re-extraction from EPUB); chunking preserved at paragraph boundaries | Chunk count and content match prior ChromaDB collection |
| 3.5.6 | Create `configs/clear_light_of_bliss.toml` capturing per-book extraction parameters | Extractor produces same output via config that the hardcoded version produced |
| 3.5.7 | End-to-end re-run on Clear Light of Bliss through the refactored pipeline | All artifacts regenerated; per-concept counts match v2.2 numbers |
| 3.5.8 | Re-run Phase 3.4a (bridge) and 3.4b (cleanup) on the fresh graph | KI-002 and KI-003 remain resolved against the new graph |
| 3.5.9 | SRENE cross-store verification | 10 random paragraphs: text in Neo4j matches text in ChromaDB chunks exactly |
| 3.5.10 | Verify KI-001 resolution end-to-end | The `clear_light + illusory_body` intersection query returns paragraphs with clean text |

### 🚦 VALIDATION GATE 3.5: Foundation Solid

All sub-phase gates pass. The system processes Clear Light of Bliss through a single canonical pipeline that is parameterized, configured, source-scoped, author-attributed, and produces clean text. The same pipeline could process book #2 by passing a different EPUB path and config — no code changes required.

### Why this comes before Phase 4

Phase 4 builds the modes (academic, meditation, eventually pastoral) on top of the retrieval foundation. Building modes on a foundation that violates S1–S6 means the modes themselves embed those violations, and the violations get expensive to remove. Phase 3.5 is the only point where the structural commitments can be made cheaply. After Phase 4 lands, refactoring becomes much harder.

---

## Phase 4: Dual-Mode RAG Integration

**Prerequisites**:
- ✅ Phase 3.4a complete (Concept→Paragraph bridge exists)
- ✅ Phase 3.4b complete (relationship vocabulary cleaned)
- ⏳ Phase 3.5 complete (structural foundation — required before Phase 4)

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
- [ ] KI-001 (text spacing) — SCHEDULED for Phase 3.5 (upstream extractor fix)
- [ ] KI-004 (self-loops) — NOTED, no current action
- [ ] KI-005 (two extraction pipelines) — SCHEDULED for Phase 3.5 (S1 resolution)
- [ ] **Phase 3.5: Structural foundation rework (S1–S6) — required before Phase 4**
- [ ] Phase 4.1: Academic query mode (Hybrid Retrieval)
- [ ] Phase 4.2: Guided meditation mode (OrganicMeditationTimer)
- [ ] Phase 4.3: Scale to full corpus (also triggers KI-001 resolution)
- [ ] Phase 4.4: Tibetan adaptation
- [x] All validation gates passed (through Phase 3.4)
- [x] "Show Me" tests answered for each phase (through Phase 3.4)


*Originally generated: January 2026 (v2.0)*
*Latest revision: June 27, 2026 (v2.3)*
*Project: Buddhist RAG System (Geshe_La_Rag)*
*Current Status: Phase 3.4 complete. Phase 3.5 (structural foundation rework, S1–S6) planned before Phase 4 begins. The system is being rebuilt as the first of 26 books, not the only one — and as a proof of concept for a wider canon-handling architecture.*
