from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
import streamlit as st
import os

# Embedding Model and Vector DB Path
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
VECTOR_DB_DIR = "vector_db"

# === Prompt & LLM Chain Setup ===
prompt = PromptTemplate(
    input_variables=["context", "q"],
    template="Answer the question based on the context:\n\n{context}\n\nQuestion: {q}"
)
llm = HuggingFaceHub(repo_id="google/flan-t5-base")
chain = LLMChain(llm=llm, prompt=prompt)

# === Load Embeddings ===
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"local_files_only": True}
    )

embedding = load_embeddings()

# === Initialize Vector Store ===
if not os.path.exists(VECTOR_DB_DIR):
    os.makedirs(VECTOR_DB_DIR)

if os.path.exists(f"{VECTOR_DB_DIR}/index.faiss"):
    vectorstore = FAISS.load_local(VECTOR_DB_DIR, embeddings=embedding, index_name="oss")
else:
    vectorstore = None


# === Store Project Metadata ===
def store_to_vector_index(project_name, metadata_dict):
    global vectorstore

    full_text = "\n".join([f"{k}: {v}" for k, v in metadata_dict.items() if isinstance(v, (str, int, float))])
    doc = Document(page_content=full_text, metadata={"project": project_name})

    if vectorstore is None:
        vectorstore = FAISS.from_documents([doc], embedding)
    else:
        vectorstore.add_documents([doc])

    vectorstore.save_local(VECTOR_DB_DIR, index_name="oss")


# === Query Vector Store ===
def query_vector_index(question: str) -> str:
    try:
        if vectorstore is None:
            return "❌ Vector index is empty. Please analyze a repository first."
        results = vectorstore.similarity_search(question, k=1)
        return results[0].page_content if results else "⚠️ No relevant match found."
    except Exception as e:
        return f"❌ Error during vector search: {str(e)}"


# === Run Prompt Chain ===
def run_chain(context_block, selected_question):
    return chain.run({"context": context_block, "q": selected_question})
