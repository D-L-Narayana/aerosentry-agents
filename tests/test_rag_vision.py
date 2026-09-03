import os

from PIL import Image

from aerosentry.rag.ingest import chunk_text, load_documents
from aerosentry.rag.store import HashEmbedder, HybridStore
from aerosentry.rag import retrieve
from aerosentry.vision import analyze, heuristic_hotspots


def test_chunking_overlap_and_bounds():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(12))
    chunks = chunk_text(text, size=300, overlap=40)
    assert len(chunks) > 3 and all(len(c) <= 300 for c in chunks)


def test_hybrid_store_ranks_relevant_chunk_first():
    s = HybridStore(HashEmbedder())
    s.add([{"id": "a", "text": "Launches are prohibited when sustained wind exceeds 10 m/s.", "source": "m"},
           {"id": "b", "text": "Propellers are replaced every 200 flight hours.", "source": "m"},
           {"id": "c", "text": "Batteries retire after 300 charge cycles.", "source": "m"}])
    hits = s.hybrid_search("what is the wind limit for a launch", k=2)
    assert hits[0]["id"] == "a"


def test_corpus_retrieval_hits_expected_manual():
    assert len(load_documents()) > 10
    hits = retrieve("fallback procedure when a model call fails", k=3)
    assert hits[0]["source"] == "safety_and_autonomy_policy.md"


def test_hotspot_detector_separates_clean_and_anomalous():
    base = os.path.join(os.path.dirname(__file__), "..", "data", "imagery")
    clean = heuristic_hotspots(Image.open(os.path.join(base, "solar_thermal_clean.png")))
    anomaly = analyze("data/imagery/solar_thermal_anomaly.png")
    hot_string = analyze("data/imagery/solar_thermal_hot_string.png")
    assert len(clean) == 0
    assert anomaly["hotspot_count"] >= 1 and anomaly["severity"] in ("medium", "high")
    assert hot_string["severity"] in ("high", "critical") and hot_string["hotspot_count"] >= 2
