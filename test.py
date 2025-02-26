from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load existing vector store (metric is already set from creation)
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
