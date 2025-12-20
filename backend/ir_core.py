from datetime import datetime
from math import exp

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

INDEX_NAME = "reuters_ir_knn"

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False
)

model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Helpers
# -------------------------
def recency_boost(date_str, decay=365):
    if not date_str:
        return 0.0
    try:
        d = datetime.fromisoformat(date_str)
        days = (datetime.now() - d).days
        return exp(-days / decay)
    except:
        return 0.0


# -------------------------
# Lexical Search
# -------------------------
def lexical_search(query, start_date=None, end_date=None, size=10):
    filters = []

    if start_date or end_date:
        r = {}
        if start_date:
            r["gte"] = start_date
        if end_date:
            r["lte"] = end_date
        filters.append({"range": {"date": r}})

    body = {
        "size": size,
        "query": {
            "bool": {
                "must": [{
                    "multi_match": {
                        "query": query,
                        "fields": ["title^5", "content"],
                        "fuzziness": "AUTO"
                    }
                }],
                "filter": filters
            }
        }
    }

    res = client.search(index=INDEX_NAME, body=body)
    return res["hits"]["hits"]


# -------------------------
# Semantic Search
# -------------------------
def semantic_search(query, size=10):
    vec = model.encode(query).tolist()

    body = {
        "size": size,
        "query": {
            "knn": {
                "content_vector": {
                    "vector": vec,
                    "k": size
                }
            }
        }
    }

    res = client.search(index=INDEX_NAME, body=body)
    return res["hits"]["hits"]


# -------------------------
# Hybrid Search
# -------------------------
def hybrid_search(query_text, start_date=None, end_date=None, size=10):
    lex = lexical_search(query_text, start_date, end_date, size * 2)
    sem = semantic_search(query_text, size * 2)

    merged = {}

    for h in lex:
        s = h["_source"]
        merged[h["_id"]] = {
            "id": h["_id"],
            "title": s["title"],
            "content": s["content"],
            "date": s.get("date"),
            "authors": s.get("authors", []),
            "locations": s.get("georeference_names", []),
            "lex": h["_score"],
            "sem": 0.0
        }

    for h in sem:
        s = h["_source"]
        if h["_id"] not in merged:
            merged[h["_id"]] = {
                "id": h["_id"],
                "title": s["title"],
                "content": s["content"],
                "date": s.get("date"),
                "authors": s.get("authors", []),
                "locations": s.get("georeference_names", []),
                "lex": 0.0,
                "sem": h["_score"]
            }
        else:
            merged[h["_id"]]["sem"] = h["_score"]

    results = []
    for d in merged.values():
        score = (
            0.6 * d["lex"]
            + 0.4 * d["sem"]
            + 0.2 * recency_boost(d["date"])
        )
        d["score"] = score
        results.append(d)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:size]


# -------------------------
# Autocomplete
# -------------------------
def autocomplete_titles(prefix, size=10):
    body = {
        "size": size,
        "_source": ["title"],
        "query": {
            "match_phrase_prefix": {
                "title": {"query": prefix}
            }
        }
    }

    res = client.search(index=INDEX_NAME, body=body)

    seen, out = set(), []
    for h in res["hits"]["hits"]:
        t = h["_source"]["title"]
        if t not in seen:
            out.append(t)
            seen.add(t)

    return out