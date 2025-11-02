import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import json
import tempfile
import os

# Set up the Streamlit app
st.set_page_config(page_title="PDF to Embeddings", page_icon="📘")
st.title("📘 PDF to Embeddings Generator (Offline)")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Function to split text into chunks
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

# Process the uploaded file
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.info(f"Reading and processing `{uploaded_file.name}`...")

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    chunks = chunk_text(text)

    # Load local embedding model
    st.info("Loading embedding model... (this may take a few seconds)")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    st.info("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    # Save data to JSON
    data = {
        "chunks": chunks,
        "embeddings": embeddings.tolist()
    }

    json_filename = uploaded_file.name.replace(".pdf", "_embeddings.json")

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Download button
    with open(json_filename, "rb") as f:
        st.success(f"✅ Generated {len(chunks)} text chunks with embeddings.")
        st.download_button(
            label="📥 Download JSON File",
            data=f,
            file_name=json_filename,
            mime="application/json"
        )

    # Cleanup
    os.remove(pdf_path)
