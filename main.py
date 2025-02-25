import yaml
import mlflow
from retrieval import retrieve_documents
from llm_model import generate_response
from mlflow_logger import log_to_mlflow
import threading

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

if __name__ == "__main__":
    while True:
        user_input = input("\nAsk something (type 'exit' to quit): ")

        if user_input.lower() == "exit":
            print("Exiting chat...")
            for thread in background_threads:
                if thread.is_alive():
                    thread.join(timeout=2.0)
            break

        with mlflow.start_run() as active_run:
            run_id = active_run.info.run_id
            result, context, cosine_score, retrieved_docs_with_scores, prompt, top_doc = process_query(user_input)

            thread = threading.Thread(
                target=log_to_mlflow, 
                args=(run_id, user_input, prompt, result, [top_doc] if top_doc else [], cosine_score)
            )
            thread.daemon = True
            thread.start()
            background_threads.append(thread)
            background_threads = [t for t in background_threads if t.is_alive()]

            print("\n🤖 **Model Response:**", result.get("response", "No response generated."))
            print("\n📜 **Top Retrieved Document Snippet:**", context[:500])
            print("🔢 **Cosine Similarity Score (Top Match):**", cosine_score)

            if retrieved_docs_with_scores:
                print("\n📚 **Retrieved Documents and Scores:**")
                for i, (doc, score) in enumerate(retrieved_docs_with_scores, 1):
                    doc_name = doc.metadata.get("source", "Unknown File")
                    print(f"{i}. 🔹 **Doc:** {doc_name[:50]}... | 🔢 **Score:** {score:.4f}")
            else:
                print("⚠️ No relevant document found.")
