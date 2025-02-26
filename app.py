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

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Create a container for the input field at the bottom
input_container = st.container()

# Display additional information in expander above the input
with st.expander("View Retrieved Documents", expanded=False):
    if len(st.session_state.messages) > 0:
        # Only show this if there's been at least one query
        last_result = st.session_state.get("last_result", {})
        last_context = st.session_state.get("last_context", "")
        last_cosine_score = st.session_state.get("last_cosine_score", 0)
        last_docs = st.session_state.get("last_docs", [])
        
        st.subheader("📜 Top Retrieved Document Snippet:")
        st.write(last_context[:500] if last_context else "No context yet")

        st.subheader("🔢 Cosine Similarity Score (Top Match):")
        st.write(round(last_cosine_score, 4) if last_cosine_score else "No score yet")

        # Show retrieved documents
        if last_docs:
            st.subheader("📚 Retrieved Documents and Scores:")
            for i, (doc, score) in enumerate(last_docs, 1):
                doc_name = doc.metadata.get("source", "Unknown File")
                st.write(f"{i}. **Doc:** {doc_name[:50]}... | **Score:** {round(score, 4)}")
        else:
            st.warning("⚠️ No relevant document found.")

# Now move to the bottom container for the input
with input_container:
    # Create a form for the chat input
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("💬 Ask Your question:", key="input", label_visibility="collapsed")
        with col2:
            submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input.strip():
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Rerun to display user message
            st.rerun()

# Process input (outside the form to avoid resubmission issues)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and "processing" not in st.session_state:
    st.session_state.processing = True
    
    # Display a spinner while processing
    with st.spinner("Thinking..."):
        # Get the last user message
        last_user_input = st.session_state.messages[-1]["content"]
        
        # Display assistant message placeholder
        with st.chat_message("assistant"):
            # Create an empty container for streaming text
            message_placeholder = st.empty()
            full_response = ""
            
            with mlflow.start_run() as active_run:
                run_id = active_run.info.run_id

                # Process query with streaming enabled
                result, context, cosine_score, retrieved_docs_with_scores, prompt, top_doc, stream_generator = process_query(last_user_input, stream=True)
                
                # Save results for the document viewer
                st.session_state.last_result = result
                st.session_state.last_context = context
                st.session_state.last_cosine_score = cosine_score
                st.session_state.last_docs = retrieved_docs_with_scores
                
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
                    args=(run_id, last_user_input, prompt, result, [top_doc] if top_doc else [], cosine_score)
                )
                thread.daemon = True
                thread.start()
                background_threads.append(thread)
    
    # Reset processing flag
    st.session_state.processing = False
    
    # Rerun to refresh UI and allow new input
    st.rerun()