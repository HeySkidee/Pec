from flask import Flask, render_template, request, send_file
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import tempfile
import json
import os

app = Flask(__name__)

# load model once (important for hosting)
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, max_length=500):
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) < max_length:
            current += p + " "
        else:
            chunks.append(current.strip())
            current = p + " "
    if current:
        chunks.append(current.strip())
    return chunks

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    pdf = request.files.get("pdf")
    if not pdf:
        return "No PDF uploaded.", 400

    # save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.save(tmp.name)
        pdf_path = tmp.name

    # extract text
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # chunk and embed
    chunks = chunk_text(text)
    embeddings = model.encode(chunks, show_progress_bar=False)

    # save output JSON
    json_filename = os.path.splitext(pdf.filename)[0] + "_embeddings.json"
    json_path = os.path.join(tempfile.gettempdir(), json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "embeddings": embeddings.tolist()}, f, ensure_ascii=False, indent=2)

    # clean temp PDF
    os.remove(pdf_path)

    return send_file(json_path, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
