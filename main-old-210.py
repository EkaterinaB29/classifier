from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd
import uuid
import numpy as np
import csv
import re
from collections import defaultdict

app = FastAPI()

MODEL_NAME = "/app/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "semantic_chunks"
VECTOR_SIZE = 384
SIMILARITY_DISTANCE = Distance.COSINE

embedder = SentenceTransformer(MODEL_NAME)
qdrant = QdrantClient("qdrant", port=6333)

# Load HS Code → Description (only 2-digit codes for broad categories)
HS_CATEGORY_MAP = {}

def load_hs_categories(file_path: str):
    try:
        df = pd.read_excel(file_path, dtype=str)
        for _, row in df.iterrows():
            raw_code = str(row.get("Code", "")).strip()
            desc = str(row.get("Description", "")).strip()
            code = re.sub(r"[^\d]", "", raw_code)
            if len(code) >= 2 and desc:
                HS_CATEGORY_MAP[code[:2].zfill(2)] = desc
    except Exception as e:
        print(f"[ERROR] Failed to load HS categories: {e}")

load_hs_categories("/app/HSCodeandDescription.xlsx")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    text: str
    score: float
    cluster_id: Optional[int] = None

def extract_hs_code(description: str) -> Optional[str]:
    match = re.search(r'HS CODE[:\s]*(\d{2,8})', description, re.IGNORECASE)
    return match.group(1) if match else None

def get_category_from_hs_code(hs_code: Optional[str]) -> str:
    if not hs_code:
        return "UNKNOWN"
    code = re.sub(r"[^\d]", "", hs_code)
    if len(code) >= 2:
        prefix = code[:2].zfill(2)
        return HS_CATEGORY_MAP.get(prefix, "UNKNOWN")
    return "UNKNOWN"

@app.post("/ingest-mounted-file")
async def ingest_mounted_file():
    file_path = "/app/containers.tsv"
    output_file_path = "/app/containers_classified.tsv"
    limit = 4000
    undefined_tokens = ["MISSING", "EMPTY", "PRAZNO", "POGREŠA"]

    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=SIMILARITY_DISTANCE)
    )

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)

    seen = {}
    unique_rows = []
    for row in rows:
        desc = row.get('CARGO_DESCRIPTION', '').strip()
        if desc not in seen and desc and len(desc) >= 5:
            seen[desc] = True
            unique_rows.append(row)
        if len(unique_rows) >= limit:
            break

    vectors, points, descs, hs_codes, row_map = [], [], [], [], {}

    for i, row in enumerate(unique_rows):
        desc = row.get('CARGO_DESCRIPTION', '').strip()
        if not desc or desc.upper() in undefined_tokens:
            row["cluster_id"] = "UNDEFINED"
            row["hs_code_assigned"] = ""
            row["hs_category"] = "UNDEFINED"
            continue
        hs_code = extract_hs_code(desc)
        category = get_category_from_hs_code(hs_code)

        # Enrich description with semantic context
        enriched_text = f"CARGO: {desc}; LOCATION: Port of Koper; CATEGORY: {category}"

        try:
            vec = embedder.encode(enriched_text, show_progress_bar=False, convert_to_numpy=True)
        except Exception as e:
            print(f"[ERROR] Embedding failed for: {desc}\n{e}")
            row["cluster_id"] = "FAILED_EMBEDDING"
            row["hs_code_assigned"] = hs_code or ""
            row["hs_category"] = category
            continue

        pid = str(uuid.uuid4())
        vectors.append(vec)
        hs_codes.append(hs_code)
        descs.append(desc)
        row_map[pid] = i
        points.append(PointStruct(
            id=pid, vector=vec.tolist(), payload={"description": desc, "hs_code": hs_code or ""}
        ))

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        vectors_np = np.array(vectors, dtype=np.float64)
        distance_matrix = cosine_distances(vectors_np).astype(np.float64)

        clustering = HDBSCAN(
            min_cluster_size=3,
            min_samples=2,
            metric='precomputed',
            cluster_selection_method='eom'
        )
        labels = clustering.fit_predict(distance_matrix)

        cluster_map = defaultdict(list)
        debug = {"labels": labels.tolist(), "num_clusters": len(set(labels)) - (1 if -1 in labels else 0)}

        for idx, label in enumerate(labels):
            pid = points[idx].id
            desc = descs[idx].lower()
            hs_code = hs_codes[idx]
            category = get_category_from_hs_code(hs_code)

            cid = str(label) if label >= 0 else f"HS_{category.replace(' ', '_').upper()}"

            qdrant.set_payload(COLLECTION_NAME, payload={"cluster_id": cid}, points=[pid])
            row = unique_rows[row_map[pid]]
            row["cluster_id"] = cid
            row["hs_code_assigned"] = hs_code or ""
            row["hs_category"] = category
            cluster_map[cid].append(desc)

    # Write output
    fieldnames = list(unique_rows[0].keys())
    if "hs_code_assigned" not in fieldnames:
        fieldnames.append("hs_code_assigned")
    if "hs_category" not in fieldnames:
        fieldnames.append("hs_category")

    with open(output_file_path, "w", encoding="utf-8", newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in unique_rows:
            writer.writerow(row)

    return {
        "status": "ok",
        "message": "Embedding + Clustering complete",
        "rows_processed": len(unique_rows),
        "clusters": {k: len(v) for k, v in cluster_map.items()},
        "debug": debug
    }

@app.get("/clusters")
async def get_clusters():
    scroll = qdrant.scroll(collection_name=COLLECTION_NAME, with_payload=True)
    clusters = {}
    for point in scroll[0]:
        cid = str(point.payload.get("cluster_id", "UNDEFINED"))
        desc = point.payload.get("description", "")
        clusters.setdefault(cid, []).append(desc)

    return {
        "clusters": {
            cid: {"cluster_id": cid, "num_items": len(txts), "examples": txts[:5]}
            for cid, txts in clusters.items()
        }
    }

@app.post("/query", response_model=List[SearchResult])
async def query(req: QueryRequest):
    return {"status": "not_implemented"}

