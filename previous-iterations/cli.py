# pdf_to_embeddings.py
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import json
import textwrap
import inquirer
import os

# 1. Ask the user to select a PDF file interactively
pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
if not pdf_files:
    print("No PDF files found in the current directory.")
    exit()

questions = [
    inquirer.List(
        'pdf_file',
        message="Which PDF file do you want to generate embeddings for?",
        choices=pdf_files
    )
]

answers = inquirer.prompt(questions)
pdf_path = answers['pdf_file']

# Show the generation message immediately after selecting the file
# print(f"You selected: {pdf_path}")
print(f"Generating embeddings for {pdf_path}...")

# 2. Read the PDF
reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

# 3. Split text into chunks (to fit within model limits)
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

# 4. Load local embedding model (no API required)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5. Generate embeddings
embeddings = model.encode(chunks, show_progress_bar=True)

# 6. Save as JSON
# Add a message to indicate the selected file
# print(f"Generating embeddings for {pdf_path}...")

# Fix the NameError by defining 'data' as the embeddings and chunks
data = {
    "chunks": chunks,
    "embeddings": embeddings.tolist()  # Convert embeddings to a list for JSON serialization
}

pdf_file_name = pdf_path.split('.')[0]  # Get the base name of the PDF file
json_file_name = f"{pdf_file_name}_embeddings.json"  # Create JSON file name

with open(json_file_name, "w", encoding="utf-8") as f:  # Use the new JSON file name
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(chunks)} text chunks with embeddings to {json_file_name}")
