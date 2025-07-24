import os
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain

# === Configurations ===
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
VECTOR_DB_DIR = "vector_db"
INDEX_NAME = "oss"

# === Prompt & LLM Chain Setup ===
prompt = PromptTemplate(
    input_variables=["context", "q"],
    template="Answer the question based on the context:\n\n{context}\n\nQuestion: {q}"
)
llm = HuggingFaceHub(repo_id="google/flan-t5-base")
chain = LLMChain(llm=llm, prompt=prompt)

# === Load Embeddings ===
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"local_files_only": True}
    )

embedding = load_embeddings()

# === Load or Initialize VectorStore ===
def load_vectorstore():
    if os.path.exists(f"{VECTOR_DB_DIR}/index.faiss"):
        return FAISS.load_local(VECTOR_DB_DIR, embeddings=embedding, index_name=INDEX_NAME)
    return None

vectorstore = load_vectorstore()

# === Store project metadata as vector ===
def store_to_vector_index(project_name: str, metadata_dict: dict):
    global vectorstore

    # Prepare metadata text
    content_lines = [f"{k}: {v}" for k, v in metadata_dict.items() if isinstance(v, (str, int, float))]
    full_text = "\n".join(content_lines)
    doc = Document(page_content=full_text, metadata={"project": project_name})

    # Create or update vectorstore
    if vectorstore is None:
        vectorstore = FAISS.from_documents([doc], embedding)
    else:
        vectorstore.add_documents([doc])

    vectorstore.save_local(VECTOR_DB_DIR, index_name=INDEX_NAME)

# === Query vector store ===
def query_vector_index(question: str) -> str:
    try:
        if vectorstore is None:
            return "❌ Vector index is empty. Please analyze a repository first."
        results = vectorstore.similarity_search(question, k=1)
        return results[0].page_content if results else "⚠️ No relevant match found."
    except Exception as e:
        return f"❌ Error during vector search: {str(e)}"

# === Run LLM QA Chain ===
def run_chain(context_block: str, question: str) -> str:
    try:
        return chain.run({"context": context_block, "q": question})
    except Exception as e:
        return f"❌ Error running the chain: {str(e)}"

# === Delete existing vector DB (if needed) ===
def delete_vector_db():
    try:
        if os.path.exists(VECTOR_DB_DIR):
            shutil.rmtree(VECTOR_DB_DIR)
            return True
        return False
    except Exception as e:
        return f"❌ Failed to delete vector DB: {str(e)}"
