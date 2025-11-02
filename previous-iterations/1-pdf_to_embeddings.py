# pdf_to_embeddings.py
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import json
import textwrap

# 1. Read the PDF
pdf_path = "pplx-at-work.pdf"
reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

# 2. Split text into chunks (to fit within model limits)
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

chunks = chunk_text(text)

# 3. Load local embedding model (no API required)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Generate embeddings
embeddings = model.encode(chunks, show_progress_bar=True)

# 5. Save as JSON
data = [{"chunk": chunks[i], "embedding": embeddings[i].tolist()} for i in range(len(chunks))]

pdf_file_name = pdf_path.split('.')[0]  # Get the base name of the PDF file
json_file_name = f"{pdf_file_name}_embeddings.json"  # Create JSON file name

with open(json_file_name, "w", encoding="utf-8") as f:  # Use the new JSON file name
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(chunks)} text chunks with embeddings to {json_file_name}")
