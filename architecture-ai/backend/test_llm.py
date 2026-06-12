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

################################# Main ######################"
if __name__ == "__main__":
    #test_chat()
    #test_stream()
    #test_base_agent()
    #test_analysis_agent()
    #test_pdf_parser()
    #test_ingestion()
    #test_retrieval()
    test_analysis_rag()