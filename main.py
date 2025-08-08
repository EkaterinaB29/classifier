from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances
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

HS_CODE_MAPPING = {
    "07": "Vegetables and Edible Plants", "08": "Fruits and Nuts",
    "09": "Coffee, Tea, Spices", "17": "Sugars and Confectionery",
    "24": "Tobacco", "27": "Mineral Fuels, Oils", "28": "Inorganic Chemicals",
    "29": "Organic Chemicals", "39": "Plastics and Articles", "40": "Rubber and Articles",
    "44": "Wood and Articles", "48": "Paper and Paperboard", "70": "Glass and Glassware",
    "72": "Iron and Steel", "73": "Articles of Iron or Steel", "76": "Aluminum and Articles",
    "84": "Machinery and Mechanical Appliances", "85": "Electrical Machinery and Equipment",
    "87": "Vehicles and Parts", "94": "Furniture, Lighting, Prefabricated Buildings"
}

KEYWORD_TO_CLUSTER = {
    "steel": "35", "iron": "35", "aluminum": "35", "furniture": "49",
    "garments": "49", "textiles": "49", "paper": "52", "plastic": "51",
    "wood": "50", "coffee": "13", "banana": "1", "solar": "19",
    "tyres": "47", "glass": "46", "machinery": "33", "vehicle parts": "32",
    "chemical": "24", "food": "41", "batteries": "14", "cheese": "11",
    "onion": "40", "potato": "39", "lamp": "21", "air conditioner": "23"
}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    text: str
    score: float
    cluster_id: Optional[int] = None

def extract_hs_code(description: str) -> Optional[str]:
    match = re.search(r'HS CODE[:\s]*(\d{6,8})', description, re.IGNORECASE)
    return match.group(1) if match else None

def get_hs_category(hs_code: Optional[str]) -> str:
    if not hs_code:
        return "UNKNOWN"
    prefix_4 = hs_code[:4]
    prefix_2 = hs_code[:2]
    return HS_CODE_MAPPING.get(prefix_4, HS_CODE_MAPPING.get(prefix_2, "UNKNOWN"))

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
            row["cluster_number"] = "UNDEFINED"
            row["cluster_label"] = "UNDEFINED"
            continue
        hs_code = extract_hs_code(desc)
        vec = embedder.encode(desc, show_progress_bar=False, convert_to_numpy=True)
        pid = str(uuid.uuid4())
        vectors.append(vec)
        hs_codes.append(hs_code)
        descs.append(desc)
        row_map[pid] = i
        points.append(PointStruct(
            id=pid, vector=vec.tolist(), payload={"description": desc, "hs_code": hs_code or ""}
        ))

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    # clustering
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
        cid = str(label) if label >= 0 else None

        if label == -1:
            for k, v in KEYWORD_TO_CLUSTER.items():
                if k in desc:
                    cid = v
                    break
            if cid is None and hs_code:
                cid = f"HS_{hs_code[:4]}_{get_hs_category(hs_code).replace(' ', '_').upper()}"
            if cid is None:
                cid = "MISCELLANEOUS"

        qdrant.set_payload(COLLECTION_NAME, payload={"cluster_id": cid}, points=[pid])
        row = unique_rows[row_map[pid]]
        row["cluster_id"] = cid
        row["cluster_number"] = cid

        if cid.startswith("HS_"):
            row["cluster_label"] = get_hs_category(hs_code)
        elif cid.isdigit():
            label_match = "UNKNOWN"
            for k, v in KEYWORD_TO_CLUSTER.items():
                if k in desc and v == cid:
                    label_match = k
                    break
            row["cluster_label"] = label_match
        else:
            row["cluster_label"] = cid

        cluster_map[cid].append(desc)

    with open(output_file_path, "w", encoding="utf-8", newline='') as out_f:
        fieldnames = list(unique_rows[0].keys())
        if "cluster_number" not in fieldnames:
            fieldnames += ["cluster_number", "cluster_label"]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in unique_rows:
            writer.writerow(row)

    return {
        "status": "ok",
        "message": "Embedding + HDBSCAN complete",
        "rows_processed": len(rows),
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

