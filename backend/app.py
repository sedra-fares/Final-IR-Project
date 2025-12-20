from flask import Flask, request, jsonify
from flask_cors import CORS
from ir_core import hybrid_search, autocomplete_titles

app = Flask(__name__)
CORS(app)


@app.route("/search", methods=["GET"])
def search():
    q = request.args.get("q", "").strip()
    start = request.args.get("from")
    end = request.args.get("to")
    size = request.args.get("size", 10, type=int)

    if not q:
        return jsonify([])

    return jsonify(
        hybrid_search(
            query_text=q,
            start_date=start,
            end_date=end,
            size=size
        )
    )


@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    q = request.args.get("q", "").strip()
    if len(q) < 3:
        return jsonify([])
    return jsonify(autocomplete_titles(q))


if __name__ == "__main__":
    print("🚀 Smart IR API running on http://127.0.0.1:5000")
    app.run(debug=True)