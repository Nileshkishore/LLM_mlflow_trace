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

# Initialize ChromaDB
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

# 🗑️ Step 1: Delete all existing documents from ChromaDB
try:
    vector_store.delete(where={"id": {"$ne": ""}})  # Correct deletion method
    print("🗑️ Deleted all existing documents from ChromaDB.")
except Exception as e:
    print(f"⚠️ Error deleting existing documents: {e}")

# Track embedding time
embedding_start_time = time.time()

# Process all text files
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        # Load document
        loader = TextLoader(file_path)
        docs = loader.load()  # Loads a list of documents

        # Ensure document has content
        if not docs or not docs[0].page_content.strip():
            print(f"⚠️ Skipping empty file: {filename}")
            continue

        # Add document to ChromaDB (Chroma automatically embeds the documents)
        vector_store.add_documents(docs)

        print(f"✅ Stored {filename}")

# Track total time
embedding_end_time = time.time()
print(f"✅ All embeddings saved to ChromaDB.")
print(f"⏱️ Total time taken: {embedding_end_time - embedding_start_time:.4f} sec")