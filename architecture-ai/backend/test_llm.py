# test_llm.py — run this directly to verify the LLM layer works
# Usage: cd backend && python test_llm.py
# Delete this file once the LLM layer is confirmed working

from app.llm.factory import get_llm
from app.llm.base import Message

def test_chat():
    print("Testing chat()...")
    llm = get_llm()
    messages = llm.build_messages(
        system_prompt="You are a helpful assistant for an architecture firm. Answer concisely.",
        user_message="What is a CCTP document in French construction? One sentence.",
    )
    response = llm.chat(messages, temperature=0.3, max_tokens=100)
    print(f"Provider : {response.provider}")
    print(f"Model    : {response.model}")
    print(f"Response : {response.content}")
    print(f"Tokens   : {response.prompt_tokens} prompt / {response.completion_tokens} completion")

def test_stream():
    print("\nTesting stream()...")
    llm = get_llm()
    messages = llm.build_messages(
        system_prompt="You are a helpful assistant. Answer concisely.",
        user_message="Say hello in French.",
    )
    print("Streamed: ", end="", flush=True)
    for token in llm.stream(messages, max_tokens=50):
        print(token, end="", flush=True)
    print()

##################### AGENT ############################
def test_base_agent():
    print("\nTesting BaseAgent...")

    # Quick concrete subclass just for testing
    from app.agents.base_agent import BaseAgent, AgentResponse

    class TestAgent(BaseAgent):
        agent_type = "analysis"
        description = "Test agent"

        def run(self, message: str) -> AgentResponse:
            return self.chat(message, max_tokens=80)

    agent = TestAgent()
    print(f"Prompt loaded: {'yes' if agent.system_prompt else 'no'}")

    response = agent.run("What is a diagnostic amiante? One sentence.")
    print(f"Agent type : {response.agent_type}")
    print(f"Duration   : {response.duration_seconds}s")
    print(f"Response   : {response.content}")


######################## Feature 1 : Analysis ##########################

def test_analysis_agent():
    print("\nTesting AnalysisAgent...")
    from app.agents.analysis_agent import AnalysisAgent

    agent = AnalysisAgent()

    # Fake document — simulates a short asbestos diagnostic
    fake_document = """
    RAPPORT DE DIAGNOSTIC AMIANTE
    Bâtiment: Immeuble Le Corbusier, 12 rue des Architectes, 75001 Paris
    Date du diagnostic: 15 mars 2024
    Diagnostiqueur: Cabinet Durand Diagnostics

    Matériaux identifiés contenant de l'amiante:
    - Flocage de la chaufferie (sous-sol): état très dégradé, intervention urgente requise
    - Dalles de sol du couloir RDC: bon état, surveillance annuelle recommandée
    - Calorifugeage des canalisations: état dégradé, confinement recommandé

    Recommandations:
    - Retrait immédiat du flocage chaufferie par entreprise certifiée
    - Surveillance annuelle des dalles de sol
    - Confinement du calorifugeage dans les 6 mois

    Prochaine visite de contrôle: mars 2025
    """

    response = agent.run(
        document_text=fake_document,
        template_name="amiante",
    )

    print(f"Agent    : {response.agent_type}")
    print(f"Duration : {response.duration_seconds}s")
    print(f"Response :\n{response.content}")

################################### PDF Parser ################################
def test_pdf_parser():
    print("\nTesting PDFParser...")
    from app.documents.parsers.pdf_parser import parse_pdf
    import sys, time


    if len(sys.argv) < 2:
        print("No PDF path provided — skipping")
        print("Usage: python test_llm.py path/to/document.pdf")
        return

    start = time.time()
    result = parse_pdf(sys.argv[1])
    duration = round(time.time() - start, 2)

    print(f"File       : {result['file_name']}")
    print(f"Pages      : {result['total_pages']} total / "
          f"{result['scanned_pages']} scanned / "
          f"{result['digital_pages']} digital")
    print(f"Words      : {result['word_count']}")
    print(f"First 500 chars:\n{result['full_text'][:500]}")
    print(f"Time       : {duration}s")


###################################### Ingesting Test ####################################
def test_ingestion():
    import sys
    import time
    print("\nTesting ingestion pipeline...")

    if len(sys.argv) < 2:
        print("No PDF path provided — skipping")
        return

    from app.documents.ingestion import ingest_document, list_documents

    # Quick chunk preview before ingesting — reads original file directly
    from app.documents.parsers.pdf_parser import parse_pdf
    from app.documents.chunker import Chunker

    print("\n--- Chunk preview ---")
    parsed = parse_pdf(sys.argv[1])
    chunker = Chunker()
    chunks = chunker.chunk_for_retrieval(parsed["full_text"])
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i:2d}: {chunk['word_count']:4d} words, {len(chunk['text']):6d} chars")
    print("--- End preview ---\n")

    # Now run the full ingestion pipeline
    start = time.time()
    metadata = ingest_document(
        file_path=sys.argv[1],
        uploaded_by="test_user",
    )
    duration = round(time.time() - start, 2)

    print(f"Doc ID     : {metadata['doc_id']}")
    print(f"File       : {metadata['file_name']}")
    print(f"Pages      : {metadata['total_pages']}")
    print(f"Words      : {metadata['word_count']}")
    print(f"Chunks     : {metadata['chunk_count']}")
    print(f"Duration   : {duration}s")
    print(f"Status     : {metadata['status']}")

    all_docs = list_documents()
    print(f"Registry   : {len(all_docs)} document(s) total")

    # Add temporarily to test_ingestion() after ingestion completes
    # import chromadb
    # client = chromadb.HttpClient(host="localhost", port=8001)
    # collection = client.get_or_create_collection("documents")
    #
    # # Count chunks stored for this document
    # results = collection.get(
    #     where={"doc_id": metadata["doc_id"]}
    # )
    # print(f"\nChromaDB verification:")
    # print(f"  Chunks stored : {len(results['ids'])}")
    # print(f"  Expected      : {metadata['chunk_count']}")
    # print(f"  Match         : {len(results['ids']) == metadata['chunk_count']}")
    #
    # # Check total characters across all chunks
    # total_chars = sum(len(doc) for doc in results['documents'])
    # print(f"  Total chars in ChromaDB : {total_chars}")
    # print(f"  Original doc chars      : {parsed['char_count']}")





###################### Retrieval #########################""""
def test_retrieval():
    print("\nTesting hybrid retrieval...")
    from app.documents.retrieval import get_retriever

    retriever = get_retriever()
    DOC_ID = "f3835063-eec5-45d3-a774-d535b73575ba"
    #DOC_ID = "bbeaddba-f734-4c8f-b5dc-d2be282c450b"

    test_queries = [
        "Quelles sont les recommandations pour la chaufferie ?",
        "Quel est l'état de conservation des matériaux ?",
        "Quelle est la date du diagnostic ?",
        "Quels matériaux contiennent de l'amiante ?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=3, doc_id=DOC_ID)

        if not results:
            print("  No results found")
            continue

        for r in results:
            print(
                f"  [hybrid:{r.hybrid_score:.0%} "
                f"sem:{r.similarity_score:.0%} "
                f"bm25:{r.bm25_score:.0%} "
                f"src:{r.source}] "
                f"chunk {r.chunk_index} — {len(r.text.split())} words"
            )
            print(f"  Preview: {r.text[:150].strip()}...")



####################### RAG Test ##########################
def test_analysis_rag():
    print("\nTesting AnalysisAgent RAG mode...")
    from app.agents.analysis_agent import AnalysisAgent

    DOC_ID = "f3835063-eec5-45d3-a774-d535b73575ba"

    agent = AnalysisAgent()
    response = agent.run_rag(
        doc_id=DOC_ID,
        template_name="amiante",
    )

    print(f"Agent    : {response.agent_type}")
    print(f"Duration : {response.duration_seconds}s")
    print(f"\n{'='*60}")
    print(response.content)
    print('='*60)






############################## Extraction agent ##############################
def test_extraction_agent():
    """
    Test script for ExtractionAgent.

    Usage:
        python test_extraction.py path/to/report.pdf --config config.xlsx [--no-cache] [--debug]

    Outputs:
        extraction_result.json        — final result
        debug_output/pass1_raw.json   — (with --debug) raw rows before dedup
        debug_output/pass2_deduped.json — (with --debug) final deduped rows
    """

    import sys
    import json
    import time
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", help="Path to PDF file")
    parser.add_argument("--config", required=True, help="Path to Excel config file (.xlsx)")
    parser.add_argument("--no-cache", action="store_true", help="Force re-parse even if cache exists")
    parser.add_argument("--debug", action="store_true", help="Write debug files")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ExtractionAgent test")
    print("=" * 60)

    # ── 0. Load config (now required — no more auto-detection mode) ─────────
    from app.documents.parsers.config_reader import ConfigReader
    print(f"\n[config] Reading {args.config}…")
    config = ConfigReader().read(args.config)
    print(f"[config] page_start    : {config.get('page_start')}")
    print(f"[config] page_end      : {config.get('page_end')}")
    print(f"[config] column_mapping: {config.get('column_mapping')}")

    if not config.get("column_mapping"):
        print("\n[config] ERROR: no column_mapping found — check the Excel "
              "'Colonne source / Colonne normalisée' table.")
        sys.exit(1)

    # ── 1. Parse PDF (or load cache) ─────────────────────────────────────────
    cache_path = Path("parsed_cache.json")

    if cache_path.exists() and not args.no_cache:
        print(f"\n[cache] Loading from {cache_path} (use --no-cache to force re-parse)")
        with open(cache_path, encoding="utf-8") as f:
            parsed = json.load(f)
        print(f"[cache] {parsed['total_pages']} pages, {parsed['word_count']} words")

    elif args.pdf:
        from app.documents.parsers.pdf_parser import parse_pdf
        print(f"\n[parse] Parsing {args.pdf}…")
        t0 = time.time()
        parsed = parse_pdf(args.pdf)
        elapsed = round(time.time() - t0, 1)
        print(f"[parse] Done in {elapsed}s — "
              f"{parsed['total_pages']} pages, {parsed['word_count']} words")
        cache_path.write_text(
            json.dumps(parsed, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[cache] Saved to {cache_path}")

    else:
        print("\nNo PDF provided and no cache found. Pass a PDF path as first argument.")
        sys.exit(1)

    # ── 2. Run extraction ─────────────────────────────────────────────────────
    from app.agents.extraction_agent import ExtractionAgent
    agent = ExtractionAgent()

    print(f"\n[extraction] Starting (debug={args.debug})…\n")
    t0 = time.time()
    response = agent.run_extraction(
        document_text=parsed["full_text"],
        config=config,
        debug=args.debug,
    )
    elapsed = round(time.time() - t0, 1)

    # ── 3. Report ─────────────────────────────────────────────────────────────
    result = json.loads(response.content)
    stats  = result["stats"]

    print(f"\n{'=' * 60}")
    print(f"  Duration        : {elapsed}s")
    print(f"  Header          : {result['header']}")
    print(f"  Total matériaux : {stats['total_materiaux']}")
    if "presence_amiante" in stats:
        print(f"  ✅ Présence      : {stats['presence_amiante']}")
        print(f"  ✅ Absence       : {stats['absence_amiante']}")
        print(f"  ⚠️  Inconnu       : {stats.get('resultat_inconnu', '—')}")
    else:
        print(f"  (pas de champ 'resultat' dans le mapping — stats présence/absence indisponibles)")

    ec = result.get("extraction_config", {})
    print(f"\n  Config used:")
    print(f"    page_start     : {ec.get('page_start')}")
    print(f"    page_end       : {ec.get('page_end')}")
    print(f"    column_mapping : {ec.get('column_mapping')}")

    if "presence_amiante" in stats:
        unknown_items = [m for m in result["materiaux"] if not m.get("resultat")]
        if unknown_items:
            print(f"\n  ⚠️  {len(unknown_items)} items without 'resultat' (first 3):")
            for m in unknown_items[:3]:
                print(f"     {m}")

    print(f"\n  First 3 items:")
    for m in result["materiaux"][:3]:
        print(f"     {m}")
    print("=" * 60)

    # ── 4. Save final result ──────────────────────────────────────────────────
    out_path = Path("extraction_result.json")
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[output] Full result → {out_path}")

    if args.debug:
        print("[output] Debug files → debug_output/pass1_raw.json")
        print("[output]               debug_output/pass2_deduped.json")




########################## Excel file test ##########################
def generate_excel_file():
    from pathlib import Path
    from app.output.excel_writer import ExcelWriter
    import json

    result_path = Path("extraction_result.json")
    if not result_path.exists():
        print("No extraction_result.json found — run extraction first.")
        return

    print("\nGenerating Excel file...")
    extraction_result = json.loads(result_path.read_text(encoding="utf-8"))
    writer = ExcelWriter()
    path = writer.write(extraction_result, output_path="rapport.xlsx")
    print(f"Excel file written to: {path}")
    print(f"  {len(extraction_result.get('materiaux', []))} rows")
    print(f"  Columns: {list(extraction_result['materiaux'][0].keys()) if extraction_result.get('materiaux') else 'none'}")






##################### RC agent test ##########################
def test_rc_agent():
    """
    Test script for RCAgent.

    Usage:
        python test_rc_agent.py path/to/rc.pdf --prompt "Insiste sur les délais" [--no-cache]
        python test_rc_agent.py path/to/rc.pdf   (no prompt → no target section hint, no emphasis)

    Outputs:
        rc_parsed_cache.json   — cached PDFParser output (skip re-parsing on reruns)
        rc_result.json         — final RCAgentResponse
    """

    import sys
    import json
    import time
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", help="Path to the RC PDF file")
    parser.add_argument("--prompt", default="", help="User prompt (section hint / emphasis)")
    parser.add_argument("--no-cache", action="store_true", help="Force re-parse even if cache exists")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("RCAgent test")
    print("=" * 60)

    # ── 1. Parse PDF (or load cache) ─────────────────────────────────────────
    cache_path = Path("rc_parsed_cache.json")

    if cache_path.exists() and not args.no_cache:
        print(f"\n[cache] Loading from {cache_path} (use --no-cache to force re-parse)")
        with open(cache_path, encoding="utf-8") as f:
            parsed = json.load(f)
        print(f"[cache] {parsed['total_pages']} pages, {parsed['word_count']} words")

    elif args.pdf:
        from app.documents.parsers.pdf_parser import parse_pdf
        print(f"\n[parse] Parsing {args.pdf}…")
        t0 = time.time()
        parsed = parse_pdf(args.pdf)
        elapsed = round(time.time() - t0, 1)
        print(f"[parse] Done in {elapsed}s — "
              f"{parsed['total_pages']} pages, {parsed['word_count']} words")
        cache_path.write_text(
            json.dumps(parsed, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[cache] Saved to {cache_path}")

    else:
        print("\nNo PDF provided and no cache found. Pass a PDF path as first argument.")
        sys.exit(1)

    # ── 2. Run RCAgent ────────────────────────────────────────────────────────
    from app.agents.rc_agent import RCAgent
    agent = RCAgent()

    print(f"\n[rc_agent] user_prompt = {args.prompt!r}")
    print("[rc_agent] Starting…\n")
    t0 = time.time()
    response = agent.run(
        document_markdown=parsed["full_text"],
        user_prompt=args.prompt,
    )
    elapsed = round(time.time() - t0, 1)
    result = response.to_dict()

    # ── 3. Report ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Duration          : {elapsed}s")
    print(f"  Model             : {result['model']}")

    intent = result["intent"]
    print(f"\n  Intent parsed:")
    print(f"    target_section_label : {intent['target_section_label']}")
    print(f"    emphasis_instructions: {intent['emphasis_instructions']}")

    structure = result["structure"]
    print(f"\n  Structure extraction:")
    if structure["found"]:
        print(f"    ✅ heading      : {structure['heading']}")
        print(f"    match_score     : {structure['match_score']}")
        print(f"\n  --- Structure ---")
        print(structure["structure"])
    else:
        print(f"    ⚠️  No matching section found (threshold not met)")

    print(f"\n  --- Summary ({len(result['summary'].split())} words) ---")
    print(result["summary"])
    print("=" * 60)

    # ── 4. Save final result ──────────────────────────────────────────────────
    out_path = Path("rc_result.json")
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[output] Full result → {out_path}")
    


################################# Main ######################"
if __name__ == "__main__":
    #test_chat()
    #test_stream()
    #test_base_agent()
    #test_analysis_agent()
    #test_pdf_parser()
    #test_ingestion()
    #test_retrieval()
    #test_analysis_rag()
    #test_extraction_agent()
    #generate_excel_file()
    test_rc_agent()
    