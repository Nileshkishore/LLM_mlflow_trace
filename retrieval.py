from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config_loader import config  # ✅ Load config from config_loader.py

# Ensure config is not None
if not config:
    raise ValueError("Configuration not loaded properly. Check config_loader.py and config.yaml.")

# Get config values
persist_directory = config["chroma"]["persist_directory"]
embedding_model = HuggingFaceEmbeddings(model_name=config["llm"]["embedding_model"])

# Initialize ChromaDB
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

def retrieve_documents(query, top_k=None):
    if top_k is None:
        top_k = config["retrieval"]["top_k"]  # Default to config value

    retrieved_docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    
    if retrieved_docs_with_scores:
        top_doc, cosine_score = retrieved_docs_with_scores[0]
        context = top_doc.page_content
    else:
        top_doc = None
        context = "No relevant document found."
        cosine_score = 0.0

    return context, cosine_score, retrieved_docs_with_scores, top_doc
