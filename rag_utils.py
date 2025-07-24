import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import Groq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Load documents ===
loader = TextLoader("sample.txt")
documents = loader.load()

# === Split documents ===
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# === Load HF Embeddings ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Create vector store ===
vectorstore = FAISS.from_documents(texts, embeddings)

# === Set up Groq LLM ===
llm = Groq(model="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY"))

# === Retrieval-based QA chain ===
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# === Ask your question ===
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == 'exit':
        break
    response = qa.run(query)
    print(f"\nAnswer: {response}")
