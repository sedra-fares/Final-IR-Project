from flask import Flask, request, jsonify
from flask_cors import CORS
from ir_core import smart_hybrid_search, autocomplete_titles, lexical_search

app = Flask(__name__)
CORS(app)

@app.route("/search", methods=["GET"])
def search():
    q = request.args.get("q", "").strip()
    start = request.args.get("from")          # YYYY-MM-DD
    end = request.args.get("to")              # YYYY-MM-DD
    geo = request.args.get("geo", "")         # Location name or "lat,lon"
    size = request.args.get("size", 10, type=int)

    if not q:
        return jsonify([])

    # Pass as tuple: (text, start_date, end_date, georeference)
    results = smart_hybrid_search((q, start, end, geo), size=size)

    # Format for frontend (add score for display)
    formatted = []
    for hit in results["hits"]["hits"]:
        source = hit["_source"]
        formatted.append({
            "id": hit["_id"],
            "title": source.get("title", "Untitled"),
            "content": source.get("content", "")[:300] + "..." if source.get("content") else "",
            "date": source.get("date"),
            "authors": source.get("authors", []),
            "locations": source.get("georeference_names", []),
            "score": hit.get("_score", 0)  # Will be custom score from re-ranking
        })

    return jsonify(formatted)

@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    q = request.args.get("q", "").strip()
    if len(q) < 3:
        return jsonify([])
    return jsonify(autocomplete_titles(q))


if __name__ == "__main__":
    print(" Smart IR API running on http://127.0.0.1:5000")
    app.run(debug=True)