# Pec - PDF to Embeddings Converter

Drag and drop a PDF file and get a JSON with text chunks and their embeddings ready to use with any LLM or RAG app.

## Run it locally

Best used offline for large PDFs. No APIs required.

### Clone:

```bash
git clone https://github.com/HeySkidee/pdf-to-embeddings-convertor
cd pdf-to-embeddings-convertor
```

### Create a virtual environment (recommended)

```
python -m venv venv
```

Activate it:

- Windows: `venv\Scripts\activate`

- Mac/Linux: `source venv/bin/activate`

### Install dependencies

```
pip install -r requirements.txt
```

### Run the app

```
python app.py
```

### Open in browser:
[http://localhost:5000](http://localhost:5000)

---

## What it does

-   Extracts text from PDFs using **PyPDF2**
-   Splits long text into readable chunks
-   Generates embeddings locally using **SentenceTransformer (all-MiniLM-L6-v2)**
-   Exports everything as a `.json` file ready for use with any LLM or RAG app

---

### Output format

```json
{
  "chunks": ["text chunk 1", "text chunk 2", "..."],
  "embeddings": [[0.12, 0.98, ...], [...]]
}
```
