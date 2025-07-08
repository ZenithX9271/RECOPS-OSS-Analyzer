# === rag_utils.py ===
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
import json

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_DIR = "vector_db"

embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# Create directory if needed
if not os.path.exists(VECTOR_DB_DIR):
    os.makedirs(VECTOR_DB_DIR)

# Lazy initialization: don't create index on empty data
if os.path.exists(f"{VECTOR_DB_DIR}/index.faiss"):
    vectorstore = FAISS.load_local(VECTOR_DB_DIR, embeddings=embedding, index_name="oss")
else:
    vectorstore = None  # Will be created when we get the first document


def store_to_vector_index(project_name, metadata_dict):
    global vectorstore  # allow reassignment

    text_blocks = []
    for key, val in metadata_dict.items():
        if isinstance(val, str):
            text_blocks.append(f"{key}: {val}")
        elif isinstance(val, (int, float)):
            text_blocks.append(f"{key}: {val}")
    full_text = "\n".join(text_blocks)

    doc = Document(page_content=full_text, metadata={"project": project_name})

    if vectorstore is None:
        # First document – create new index
        vectorstore = FAISS.from_documents([doc], embedding)
    else:
        # Add to existing index
        vectorstore.add_documents([doc])

    vectorstore.save_local(VECTOR_DB_DIR, index_name="oss")


def query_vector_index(question: str) -> str:
    try:
        if vectorstore is None:
            return "❌ Vector index is empty. Please analyze a repository first."
        results = vectorstore.similarity_search(question, k=1)
        return results[0].page_content if results else "⚠️ Sorry, no relevant match found in analyzed data."
    except Exception as e:
        return f"❌ Error during vector search: {str(e)}"
