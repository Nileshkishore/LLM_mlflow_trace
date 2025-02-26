import yaml
import mlflow
import streamlit as st
import threading
from retrieval import retrieve_documents
from llm_model import generate_response, generate_response_stream
from mlflow_logger import log_to_mlflow
from config_loader import config


mlflow.set_experiment(config["mlflow"]["experiment_name"])

# Keep track of background threads
background_threads = []

@mlflow.trace
def process_query(user_input, stream=False):
    # Retrieve documents
    context, cosine_score, retrieved_docs_with_scores, top_doc = retrieve_documents(user_input)
    
    if top_doc.metadata["source"] == "00-Sports-Articles/vulgar.txt":
        result = {"response": "no comment"}
        prompt = f"Question: {user_input}"
        return result, context, cosine_score, retrieved_docs_with_scores, prompt, top_doc, None
    else:
        # Construct the full prompt
        if cosine_score <= config["score"]["thresold"]:
            prompt = f"Context: {context}\n\nQuestion: {user_input}"
        else:
            prompt = f"Question: {user_input}"

        if stream:
            # Return the generator for streaming
            stream_generator = generate_response_stream(prompt)
            # For MLflow logging later
            result = {"response": ""}  # Will be filled during streaming
            return result, context, cosine_score, retrieved_docs_with_scores, prompt, top_doc, stream_generator
        else:
            # Query Llama 3.2 with retrieved context (non-streaming)
            result = generate_response(prompt)
            return result, context, cosine_score, retrieved_docs_with_scores, prompt, top_doc, None

# Streamlit UI
st.title("🔍 LLM RAG Chatbot with MLflow & ChromaDB")
st.write("Ask a question and get responses from Llama 3.2!")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
user_input = st.text_input("💬 Ask Your question:", "")

if st.button("Submit"):
    if user_input.strip():
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Display assistant message placeholder
        with st.chat_message("assistant"):
            # Create an empty container for streaming text
            message_placeholder = st.empty()
            full_response = ""
            
            with mlflow.start_run() as active_run:
                run_id = active_run.info.run_id

                # Process query with streaming enabled
                result, context, cosine_score, retrieved_docs_with_scores, prompt, top_doc, stream_generator = process_query(user_input, stream=True)
                
                # If streaming is available (non-vulgar content)
                if stream_generator:
                    # Stream the response
                    for token in stream_generator:
                        full_response += token
                        message_placeholder.markdown(full_response)
                    
                    # Update result for MLflow logging
                    result["response"] = full_response
                else:
                    # Display non-streaming response
                    message_placeholder.markdown(result["response"])
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": result["response"]})
                
                # Log asynchronously
                thread = threading.Thread(
                    target=log_to_mlflow, 
                    args=(run_id, user_input, prompt, result, [top_doc] if top_doc else [], cosine_score)
                )
                thread.daemon = True
                thread.start()
                background_threads.append(thread)

            # Display additional information
            with st.expander("View Retrieved Documents"):
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