import fitz  
import re
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct


load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "bio11"  
PDF_PATH = "data/Class-XI-Biology.pdf" 


def clean_text(raw):
    
    clean = re.sub(r'[^\x00-\x7F]+', ' ', raw)
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()


def extract_chunks_from_pdf(pdf_path, chunk_size=200, overlap=50):
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text("text") for page in doc)
    cleaned_text = clean_text(full_text)
    words = cleaned_text.split()

    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def upload_to_qdrant(chunks):
    print(f"✅ Connecting to Qdrant Cloud at {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        print("📁 Creating collection...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"🔢 Encoding {len(chunks)} chunks...")

    points = []
    for idx, chunk in enumerate(chunks):
        vector = model.encode(chunk)
        point = PointStruct(
            id=idx,
            vector=vector,
            payload={"text": chunk}
        )
        points.append(point)

    print("📤 Uploading to Qdrant...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ Upload complete.")


if __name__ == "__main__":
    print("📘 Loading and chunking PDF...")
    chunks = extract_chunks_from_pdf(PDF_PATH)
    print(f"✅ Extracted {len(chunks)} clean chunks.")
    upload_to_qdrant(chunks)
