import os
import re
import time
from datetime import datetime
from bs4 import BeautifulSoup

import spacy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from dateparser.search import search_dates

from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================
DATA_PATH = "reuters21578"
INDEX_NAME = "reuters_ir_knn"
# =========================================

# OpenSearch
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False
)

# Models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_md")
geolocator = Nominatim(user_agent="reuters_ir")

GEO_CACHE = {}

# -------------------------
# Utilities
# -------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def embed(text):
    return embedder.encode(text).tolist()


def geocode_cached(name, retries=3):
    key = name.lower().strip()
    if key in GEO_CACHE:
        return GEO_CACHE[key]

    for _ in range(retries):
        try:
            loc = geolocator.geocode(name, timeout=10)
            if loc:
                GEO_CACHE[key] = {"lat": loc.latitude, "lon": loc.longitude}
                return GEO_CACHE[key]
        except (GeocoderTimedOut, GeocoderUnavailable):
            time.sleep(1)

    GEO_CACHE[key] = None
    return None


def extract_georeferences(text, sgml_places):
    names = set(sgml_places or [])
    points = []

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            names.add(ent.text)

    for name in names:
        loc = geocode_cached(name)
        if loc:
            points.append(loc)

    return list(names), points


def extract_temporal_expressions(text):
    found = search_dates(
        text,
        settings={"PREFER_DATES_FROM": "past"}
    )
    if not found:
        return []
    return list({d.isoformat() for _, d in found if d})


# -------------------------
# Reuters SGML Loader
# -------------------------
def load_documents(path):
    for fname in os.listdir(path):
        if not fname.endswith(".sgm"):
            continue

        with open(os.path.join(path, fname), encoding="latin1") as f:
            soup = BeautifulSoup(f.read(), "lxml")

        for r in soup.find_all("reuters"):
            title = r.title.text if r.title else ""
            body = r.body.text if r.body else ""
            if not body:
                continue

            date = None
            if r.date:
                try:
                    date = datetime.strptime(r.date.text.strip(), "%d-%b-%Y")
                except:
                    pass

            places = [p.text for p in r.find_all("places")]
            authors = [a.text for a in r.find_all("author")]

            yield title, body, date, authors, places


# -------------------------
# Index Creation
# -------------------------
def create_index():
    if client.indices.exists(INDEX_NAME):
        client.indices.delete(INDEX_NAME)

    mapping = {
        "settings": {
            "index": {"knn": True}
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "content": {"type": "text"},
                "content_vector": {
                    "type": "knn_vector",
                    "dimension": 384
                },
                "date": {"type": "date"},
                "authors": {
    "type": "object", # Change 'nested' to 'object'
    "properties": {
        "first_name": {"type": "text"},
        "last_name": {"type": "text"},
        "email": {"type": "keyword"}
    }
},
                "georeference_names": {"type": "keyword"},
                "geopoint": {"type": "geo_point"},
                "temporal_expressions": {"type": "date"}
            }
        }
    }

    client.indices.create(index=INDEX_NAME, body=mapping)


# -------------------------
# Bulk Index
# -------------------------
def bulk_index():
    actions = []

    for title, body, date, authors, places in load_documents(DATA_PATH):
        text = clean_text(title + " " + body)

        geo_names, geo_points = extract_georeferences(text, places)
        temporal = extract_temporal_expressions(text)

        actions.append({
            "_index": INDEX_NAME,
            "_source": {
                "title": title,
                "content": body,
                "content_vector": embed(body),
                "date": date.isoformat() if date else None,
                "authors": authors,
                "georeference_names": geo_names,
                "geopoint": geo_points[0] if geo_points else None,
                "temporal_expressions": temporal
            }
        })

    helpers.bulk(client, actions)


if __name__ == "__main__":
    create_index()
    bulk_index()
    print("✅ Indexing complete")