import yaml
import mlflow
import streamlit as st
import threading
from retrieval import retrieve_documents
from llm_model import generate_response
from mlflow_logger import log_to_mlflow

# Load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

mlflow.set_experiment(config["mlflow"]["experiment_name"])

# Keep track of background threads
background_threads = []
@mlflow.trace
def process_query(user_input):
    # Retrieve documents
    context, cosine_score, retrieved_docs_with_scores, top_doc = retrieve_documents(user_input)

    # Construct the full prompt
    prompt = f"Context: {context}\n\nQuestion: {user_input}"

    # Query Llama 3.2 with retrieved context
    result = generate_response(prompt)

    return result, context, cosine_score, retrieved_docs_with_scores, prompt, top_doc

# Streamlit UI
st.title("🔍 LLM RAG Chatbot with MLflow & ChromaDB")
st.write("Ask a question and get responses from Llama 3.2!")

# User input
user_input = st.text_input("💬 Ask Your question:", "")

if st.button("Submit"):
    if user_input.strip():
        with mlflow.start_run() as active_run:
            run_id = active_run.info.run_id

            # Process query
            result, context, cosine_score, retrieved_docs_with_scores, prompt, top_doc = process_query(user_input)

            # Log asynchronously
            thread = threading.Thread(
                target=log_to_mlflow, 
                args=(run_id, user_input, prompt, result, [top_doc] if top_doc else [], cosine_score)
            )
            thread.daemon = True
            thread.start()
            background_threads.append(thread)

            # Display response
            st.subheader("🤖 Model Response:")
            st.write(result.get("response", "No response generated."))

            st.subheader("📜 Top Retrieved Document Snippet:")
            st.write(context[:500])

            st.subheader("🔢 Cosine Similarity Score (Top Match):")
            st.write(round(cosine_score, 4))

            # Show retrieved documents
            if retrieved_docs_with_scores:
                st.subheader("📚 Retrieved Documents and Scores:")
                for i, (doc, score) in enumerate(retrieved_docs_with_scores, 1):
                    doc_name = doc.metadata.get("source", "Unknown File")
                    st.write(f"{i}. **Doc:** {doc_name[:50]}... | **Score:** {round(score, 4)}")
            else:
                st.warning("⚠️ No relevant document found.")
    else:
        st.error("❌ Please enter a question.")
