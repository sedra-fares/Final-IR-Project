from datetime import datetime
from math import exp
from datetime import timedelta
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from geopy.distance import geodesic
from indexer import  geocode_cached
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
def smart_hybrid_search(
    query_tuple,  # Tuple: (query_text, start_date, end_date, georeference)
    size: int = 10
 ):
    """
    Advanced hybrid search with:
    - Lexical + semantic retrieval
    - Date range filtering
    - Geo proximity boost
    - Title match boost
    - Recency boost
    - Final manual re-ranking
    """
    if not isinstance(query_tuple, tuple) or len(query_tuple) != 4:
        raise ValueError("Query must be a tuple: (query_text, start_date, end_date, georeference)")

    query_text, start_date_input, end_date_input, georef = query_tuple

    # Convert inputs to datetime
    def to_datetime(obj):
        if obj is None:
            return None
        if isinstance(obj, str):
            return datetime.fromisoformat(obj.split('T')[0])
        if isinstance(obj, (datetime, datetime.date)):
            return datetime.combine(obj.date() if hasattr(obj, 'date') else obj, datetime.min.time())
        return None

    start = to_datetime(start_date_input)
    end = to_datetime(end_date_input)

    # Default end date to Dec 31 of year if only start provided
    if end is None and start:
        end = datetime(start.year, 12, 31)

    # Build date range filter
    date_range = {}
    if start:
        date_range["gte"] = start.date().isoformat()
    if end:
        date_range["lte"] = (end.date() + timedelta(days=1) - timedelta(seconds=1)).isoformat()
    range_filter = {"range": {"date": date_range}} if date_range else None

    # Resolve georeference to coordinates
    lat = lon = None
    if georef and str(georef).strip():
        g = str(georef).strip()
        if "," in g:
            try:
                lat, lon = map(float, g.split(","))
            except:
                pass
        if lat is None:
            loc = geocode_cached(g)  # Make sure this function is available
            if loc:
                lat, lon = loc["lat"], loc["lon"]
            elif "usa" in g.lower():
                lat, lon = 39.8283, -98.5795

    query_point = (lat, lon) if lat is not None and lon is not None else None

    # 1. Lexical search
    lex_body = {
        "size": size * 5,
        "query": {
            "bool": {
                "must": [{"multi_match": {"query": query_text, "fields": ["title^5", "content"]}}],
                "filter": [range_filter] if range_filter else []
            }
        }
    }
    lex_hits = client.search(index=INDEX_NAME, body=lex_body)["hits"]["hits"]

    # 2. Semantic search
    query_vector = model.encode(query_text).tolist()
    sem_body = {
        "size": size * 10,
        "query": {"knn": {"content_vector": {"vector": query_vector, "k": size * 10}}}
    }
    sem_hits = client.search(index=INDEX_NAME, body=sem_body)["hits"]["hits"]

    # 3. Combine candidates
    candidates = {}
    for hit in lex_hits:
        doc_id = hit["_id"]
        candidates[doc_id] = {
            "hit": hit,
            "source": hit["_source"],
            "score": hit["_score"],
            "lexical_score": hit["_score"],
            "semantic_score": 0.0
        }

    for hit in sem_hits:
        source = hit["_source"]
        date_str = source.get("date")
        # Apply date filter to semantic results
        if date_range and date_str:
            try:
                doc_date = datetime.fromisoformat(date_str.replace("Z", "")).date()
                if start and doc_date < start.date():
                    continue
                if end and doc_date > end.date():
                    continue
            except:
                continue
        elif date_range and not date_str:
            continue

        doc_id = hit["_id"]
        if doc_id in candidates:
            candidates[doc_id]["semantic_score"] = hit["_score"]
            candidates[doc_id]["score"] += hit["_score"]
        else:
            candidates[doc_id] = {
                "hit": hit,
                "source": source,
                "score": hit["_score"],
                "lexical_score": 0.0,
                "semantic_score": hit["_score"]
            }

    # 4. Re-ranking
    now = datetime(1987, 12, 31)  # Corpus year
    query_words = set(query_text.lower().split())

    ranked = []
    for info in candidates.values():
        source = info["source"]
        final_score = info["score"]

        # Title boost
        title_words = set(source.get("title", "").lower().split())
        matches = len(query_words.intersection(title_words))
        if matches > 0:
            final_score *= (1 + matches * 2.5)

        # Recency boost
        date_str = source.get("date")
        if date_str:
            try:
                doc_date = datetime.fromisoformat(date_str.replace("Z", ""))
                months_old = max(0, (now - doc_date).days // 30)
                final_score *= max(0.5, 1 - months_old * 0.02)
            except:
                pass

        # Geo proximity boost
        if query_point:
            gp = source.get("geopoint", {})
            if gp.get("lat") and gp.get("lon"):
                try:
                    dist = geodesic(query_point, (gp["lat"], gp["lon"])).km
                    final_score *= max(0.1, 10 ** (-dist / 10000))
                except:
                    pass

        ranked.append((info["hit"], final_score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    top_hits = []
    for hit, final_score in ranked[:size]:
       hit["_score"] = round(final_score, 3)  # Show real hybrid score
       top_hits.append(hit)
    return {"hits": {"hits": top_hits}}
# -------------------------
# Autocomplete
# -------------------------


def autocomplete_titles(prefix, size=10):
    if len(prefix) < 3:
        return []
    body = {
        "_source": ["title"],
        "size": size,
        "query": {
            "bool": {
                "should": [
                    {"match_phrase_prefix": {"title": {"query": prefix}}},
                    {"match": {"title": {"query": prefix, "fuzziness": 1, "prefix_length": 1}}}
                ]
            }
        }
    }
    res = client.search(index=INDEX_NAME, body=body)
    seen = set()
    titles = []
    for hit in res["hits"]["hits"]:
        t = hit["_source"].get("title","").strip()
        if t and t not in seen:
            titles.append(t)
            seen.add(t)
        if len(titles) >= size:
            break
    return titles    



'''def spatio_temporal_search(
        text: str = None,
        start: str = None,
        end: str = None,
        country: str = None,
        lat: float = None,
        lon: float = None,
        radius_km: int = 50,
        size: int = 10
        ):
       

    filters = []
    # Date range filter
    if start or end:
     rng = {}
    if start:
        rng["gte"] = start
    if end:
        rng["lte"] = end
    filters.append({"range": {"date": rng}})

    # Country filter (exact match; if variants are an issue, normalize queries or use match with analyzer)
    if country:
     filters.append({"term": {"georeference_names.keyword": country}})

    # Geo distance filter
    if lat is not None and lon is not None:
     filters.append({
        "geo_distance": {
            "distance": f"{radius_km}km",
            "geopoint": {"lat": lat, "lon": lon}
        }
    })

    # Base queries: lexical + semantic (if text provided)
    base_should = []
    if text:
     base_should.append({"multi_match": {"query": text, "fields": ["title^5", "content"]}})
    query_vector = [float(x) for x in get_embedding_cached(text)]
    base_should.append({
        "knn": {
            "field": "content_vector",
            "query_vector": query_vector,
            "k": size,
            "num_candidates": 100
        }
    })

    # If no text, default to match_all
    if not base_should:
     base_should.append({"match_all": {}})
    
     functions = [
            {
                "gauss": {
                    "date": {
                        "origin": "now",
                        "scale": "365d",
                        "decay": 0.9
                    }
                }
            }
        ]
     if lat is not None and lon is not None:
            functions.append({
                "gauss": {
                    "geopoint": {
                        "origin": {"lat": lat, "lon": lon},
                        "scale": "2000km",
                        "decay": 0.8
                    }
                }
            })

   
     body = {
            "size": size,
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "should": base_should,
                            "filter": filters if filters else []  # Empty filters mean no filtering
                        }
                    },
                    "functions": functions,
                    "boost_mode": "multiply",
                    "score_mode": "multiply"
                }
            },
            "_source": ["title", "content", "date", "georeference_names", "geopoint"]
        }

     return client.search(index=INDEX_NAME, body=body)
'''

def analytics_top_georefs_and_timeline(top_n=10):
    body = {
        "size": 0,
        "aggs": {
            "top_places": {
                "terms": {
                    "field": "georeference_names.keyword",
                    "size": top_n,
                    "order": {"_count": "desc"}
                }
            },
            "timeline": {
                "date_histogram": {
                    "field": "date",
                    "calendar_interval": "day",
                    "format": "yyyy-MM-dd"
                }
            }
        }
    }
    return client.search(index=INDEX_NAME, body=body)