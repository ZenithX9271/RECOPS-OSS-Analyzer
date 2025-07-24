# === FILE: rag_utils.py ===
import os
import shutil
import pickle
from sentence_transformers import SentenceTransformer
import faiss

# === Configuration ===
EMBED_MODEL_NAME = "paraphrase-MiniLM-L6-v2"
VECTOR_DB_DIR = "vector_db"
INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss.index")
METADATA_FILE = os.path.join(VECTOR_DB_DIR, "metadata.pkl")

# === Initialize model ===
model = SentenceTransformer(EMBED_MODEL_NAME)

# === Ensure directory ===
if not os.path.exists(VECTOR_DB_DIR):
    os.makedirs(VECTOR_DB_DIR)

# === Load or initialize index and metadata ===
if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
else:
    index = None
    metadata = []

# === Store project metadata into vector index ===
def store_to_vector_index(project_name: str, metadata_dict: dict):
    global index, metadata
    # Prepare text content for embedding
    content = "\n".join(f"{k}: {v}" for k, v in metadata_dict.items() if isinstance(v, (str, int, float)))
    # Compute embedding
    vec = model.encode([content])
    # Initialize index if needed
    if index is None:
        dim = vec.shape[1]
        index = faiss.IndexFlatL2(dim)
    # Add to index and metadata list
    index.add(vec)
    metadata.append((project_name, content))
    # Save index and metadata
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

# === Query vector store ===
def query_vector_index(question: str) -> str:
    if index is None or index.ntotal == 0:
        return "❌ Vector index is empty. Please analyze a repository first."
    # Embed question
    qvec = model.encode([question])
    # Search nearest neighbor
    distances, indices = index.search(qvec, k=1)
    idx = indices[0][0]
    proj_name, content = metadata[idx]
    return content
