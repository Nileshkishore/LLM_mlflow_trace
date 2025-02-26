import os
import time
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Disable parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define paths
folder_path = "00-Sports-Articles"
persist_directory = "./chroma_db"

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 1: Clear existing directory (optional: comment out if you want to keep a backup)
if os.path.exists(persist_directory):
    import shutil
    shutil.rmtree(persist_directory)
    print(f"🗑️ Cleared existing {persist_directory} directory.")

# Step 2: Load all documents first
docs = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        loader = TextLoader(file_path)
        loaded_docs = loader.load()
        
        if not loaded_docs or not loaded_docs[0].page_content.strip():
            print(f"⚠️ Skipping empty file: {filename}")
            continue
        docs.extend(loaded_docs)
        print(f"📜 Loaded {filename}")

# Step 3: Create new ChromaDB with cosine similarity
embedding_start_time = time.time()
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"}  # Enforce cosine similarity
)
# No need for persist() - it’s automatic with persist_directory

embedding_end_time = time.time()
print(f"✅ All embeddings saved to ChromaDB with cosine similarity.")
print(f"⏱️ Total time taken: {embedding_end_time - embedding_start_time:.4f} sec")

# Step 4: Test it
query = "football game recap"
results = vector_store.similarity_search_with_score(query, k=4)
for doc, score in results:
    print(f"Doc: {doc.page_content[:50]}... | Score: {score}")