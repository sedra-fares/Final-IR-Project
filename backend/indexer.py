import os
import re
import time
from datetime import datetime
from bs4 import BeautifulSoup
import dateparser
import spacy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from dateparser.search import search_dates
from geopy.distance import geodesic
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================
DATA_PATH = r"C:\Users\User\OneDrive\Desktop\Final-IR-Project\Final-IR-Project\data\reut2-0000.sgm"
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
    if not text.strip():
        return [0.0] * 384
    return embedder.encode(text).tolist()

def geocode_cached(name, retries=3):
    if not name:
        return None
    key = name.lower().strip()
    if key in GEO_CACHE:
        return GEO_CACHE[key]

    result = None
    for _ in range(retries):
        try:
            loc = geolocator.geocode(name, timeout=10)
            if loc:
                # Store the FULL location object from Nominatim
                full_result = {
                    "lat": loc.latitude,
                    "lon": loc.longitude,
                    "address": loc.address,
                    "display_name": loc.raw.get("display_name"),
                    "raw": loc.raw  # Full JSON response from Nominatim
                }
                result = full_result
                break
        except (GeocoderTimedOut, GeocoderUnavailable):
            time.sleep(1)
        except Exception as e:
            print(f"Geocoding error for '{name}': {e}")
            break

    GEO_CACHE[key] = result
    return result  # Returns full dict or None


def extract_georeferences(text, sgml_places):

    names = set(sgml_places or [])
    additional_names = set()  # Collect new countries here
    points = []

    # spaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            name = ent.text.strip()
            if len(name) < 3 or name.isdigit() or re.search(r"\d", name):
                continue
            names.add(name)

    # Geocode and extract countries without modifying during iteration
    for name in list(names):  # Use list() to make a copy for safe iteration
        loc = geocode_cached(name)
        if loc:
            lat, lon = loc["lat"], loc["lon"]
            points.append({"lat": lat, "lon": lon})

            # Reverse geocode to get country
            try:
                reverse_loc = geolocator.reverse((lat, lon), language="en", timeout=10)
                if reverse_loc and reverse_loc.raw.get("address"):
                    country = reverse_loc.raw["address"].get("country")
                    if country:
                        additional_names.add(country)
            except (GeocoderTimedOut, GeocoderUnavailable):
                pass
            except Exception as e:
                print(f"Reverse geocode error for {name}: {e}")

    # Now safely add the new countries
    names.update(additional_names)

    return list(names), points

def extract_temporal_expressions(text):
    found = search_dates(text, settings={"PREFER_DATES_FROM": "past"})
    if not found:
        return []
    return [d.isoformat() for _, d in found if d]

# -------------------------
# Reuters SGML Loader (Single File)
# -------------------------
def load_documents(single_file_path):
    print(f"Loading single file: {single_file_path}")
    
    with open(single_file_path, "r", encoding="latin-1", errors="ignore") as f:
        soup = BeautifulSoup(f, "lxml")
    
    print(f"Found {len(soup.find_all('reuters'))} <REUTERS> articles in the file")
    
    doc_counter = 1
    for reuters in soup.find_all("reuters"):
        doc_id = str(doc_counter)
        doc_counter += 1

        # Title
        title_tag = reuters.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Text and Author
        text_tag = reuters.find("text")
        authors = []
        body = ""

        author_tag = text_tag.find("author") if text_tag else None

        # === FULL AUTHOR PARSING ===
        if author_tag and author_tag.get_text(strip=True):
            author_text = author_tag.get_text(strip=True).strip()

            cleaned = author_text
            if cleaned.lower().startswith("by "):
                cleaned = cleaned[3:].strip()

            cleaned = re.sub(r",\s*Reuters\)?$", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*\(Reuters\)$", "", cleaned, flags=re.IGNORECASE)

            email = ""
            email_match = re.search(r'[<\(]([^@\s]+@[^@\s\)>]+)[>\)]', cleaned)
            if email_match:
                email = email_match.group(1).strip().lower()
                cleaned = re.sub(r'[<\(][^@\s]+@[^@\s\)>]+[>\)]', '', cleaned).strip()

            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

            name_parts = cleaned.split()
            if len(name_parts) >= 2:
                first_name = name_parts[0]
                last_name = " ".join(name_parts[1:])
            elif len(name_parts) == 1:
                first_name = name_parts[0]
                last_name = ""
            else:
                first_name = "Unknown"
                last_name = "Author"

            authors.append({
                "first_name": first_name,
                "last_name": last_name,
                "email": email
            })
        else:
            # Fallback for no author
            authors.append({
                "first_name": "Unknown",
                "last_name": "Author",
                "email": ""
            })

        # === BODY EXTRACTION ===
        if text_tag:
            if author_tag:
                author_tag.decompose()

            raw_text = text_tag.get_text(separator=" ", strip=False)
            cleaned_body = re.sub(r'&#\d+;', ' ', raw_text)
            cleaned_body = re.sub(r'\bRM\b|\bf\d{4}\b|\breute\b', ' ', cleaned_body, flags=re.IGNORECASE)
            cleaned_body = re.sub(r'\s+', ' ', cleaned_body)
            cleaned_body = re.sub(r'\s+REUTER[S]?\s*$', '', cleaned_body, flags=re.IGNORECASE)
            body = cleaned_body.strip()

            if len(body.split()) < 5:
                continue

        if not body:
            continue

        # === DATE ===
        explicit_date = None
        date_tag = reuters.find("date")
        if date_tag and date_tag.get_text(strip=True):
            explicit_date = dateparser.parse(
                date_tag.get_text(strip=True),
                settings={"PREFER_DATES_FROM": "past", "RELATIVE_BASE": datetime(1987, 1, 1)}
            )

        # === PLACES ===
        places = []
        places_tag = reuters.find("places")
        if places_tag:
            for d in places_tag.find_all("d"):
                txt = d.get_text(strip=True)
                if txt:
                    places.append(txt)

        yield doc_id, title, body, authors, explicit_date, places

# -------------------------
# Index Creation
# -------------------------
def create_index():
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index: {INDEX_NAME}")

    mapping = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index": {"knn": True},
        "analysis": {
            "filter": {
                "edge_ngram_filter": {"type": "edge_ngram", "min_gram": 3, "max_gram": 20},
                "english_stop": {"type": "stop", "stopwords": "_english_"},
                "english_stemmer": {"type": "stemmer", "name": "english"}
            },
            "analyzer": {
                "autocomplete": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "edge_ngram_filter"]
                },
                "content_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "english_stop", "english_stemmer"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "title": {"type": "text", "analyzer": "autocomplete", "search_analyzer": "standard"},
            "content": {"type": "text", "analyzer": "content_analyzer"},
            "content_vector": {"type": "knn_vector", "dimension": 384},
            "authors": {"type": "nested", "properties": {
                "first_name": {"type": "text"},
                "last_name": {"type": "text"},
                "email": {"type": "keyword"}
            }},
            "date": {"type": "date"},
            "geopoint": {"type": "geo_point"},
            "temporal_expressions": {"type": "date"},
            "georeferences": {"type": "geo_point"},
            "georeference_names": {"type": "keyword"},
            "original_sgml_places": {"type": "keyword"}
        }
    }
}
    client.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Created index: {INDEX_NAME}")

def doc_to_action(doc_id, title, body, authors, explicit_date, places):
    full_text = (title or "") + " " + (body or "")
    
    cleaned_content = clean_text(body)
    vector = embed(cleaned_content)

    temporal = extract_temporal_expressions(full_text)

    date_val = explicit_date.isoformat() if explicit_date else None
    if not date_val:
        valid_dates = [d for d in temporal if d]
        if valid_dates:
            date_val = min(datetime.fromisoformat(d) for d in valid_dates).isoformat()

    # Use SGML <PLACES> if available (official, curated)
    if places:
     # Only use the official places from SGML
     geo_names = list(set(places))  # Dedupe and keep order-ish
     # Geocode only the official places
     geo_points = []
     for name in geo_names:
        loc = geocode_cached(name)
        if loc:
            geo_points.append({"lat": loc["lat"], "lon": loc["lon"]})
    else:
     # No <PLACES> tag → fall back to full text extraction (spaCy + reverse)
     geo_names, geo_points = extract_georeferences(full_text, [])
    
    if not geo_names:
        geo_names = ['UNKNOWN']
    
    if not geo_points:
        geo_points = [{"lat": 0.0, "lon": 0.0}]
    
    geopoint = geo_points[0]

    # Use the already-structured authors from load_documents
    # Ensure there's always at least one
    if not authors:
        authors = [{
            "first_name": "Unknown",
            "last_name": "Author",
            "email": ""
        }]

    source = {
        "title": title,
        "content": body,
        "content_vector": vector,
        "authors": authors,
        "date": date_val,
        "temporal_expressions": temporal,
        "georeferences": geo_points,
        "georeference_names": geo_names,
        "geopoint": geopoint,
        "original_sgml_places": places
    }

    return {
        "_op_type": "index",
        "_index": INDEX_NAME,
        "_id": doc_id,
        "_source": source
    }
# -------------------------
# Bulk Index
# -------------------------
def bulk_index():
    create_index()
    actions = []
    total = 0

    for doc_id, title, body, authors, explicit_date, places in load_documents(DATA_PATH):
        action = doc_to_action(doc_id, title, body, authors, explicit_date, places)
        actions.append(action)

        if len(actions) >= 300:
            success, _ = bulk(client, actions)
            total += success
            print(f"Indexed {success} documents, total: {total}")
            actions = []

    if actions:
        success, _ = bulk(client, actions)
        total += success
        print(f"Final batch: {success} documents, total: {total}")

    print(f"Indexing complete! Total indexed: {total}")

if __name__ == "__main__":
    bulk_index()