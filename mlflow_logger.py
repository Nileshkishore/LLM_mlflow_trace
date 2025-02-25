import mlflow
import threading

def log_to_mlflow(run_id, user_input, prompt, result, retrieved_docs, cosine_score):
    def log():
        with mlflow.start_run(run_id=run_id):
            mlflow.log_param("model_used", result.get("model", "Unknown Model"))
            mlflow.log_param("user_prompt", user_input)
            #mlflow.log_param("full_prompt", prompt)

            retrieved_doc_name = retrieved_docs[0].metadata.get("source", "Unknown File") if retrieved_docs else "No document found"
            mlflow.log_param("retrieved_doc_name", retrieved_doc_name)

            mlflow.log_metric("cosine_similarity", cosine_score)
            mlflow.log_metric("processing_time_us", result.get("total_duration", 0))

            # Token-related metrics
            prompt_tokens = result.get("prompt_tokens", 0)
            generated_tokens = result.get("generated_tokens", 0)

            mlflow.log_metric("prompt_tokens", prompt_tokens)
            mlflow.log_metric("generated_tokens", generated_tokens)

            # Cost calculations
            input_cost = (prompt_tokens / 1_000) * 0.003
            output_cost = (generated_tokens / 1_000) * 0.015
            total_cost = input_cost + output_cost

            mlflow.log_metric("input_cost_usd", round(input_cost, 6))
            mlflow.log_metric("output_cost_usd", round(output_cost, 6))
            mlflow.log_metric("total_cost_usd", round(total_cost, 6))

            llm_response = result.get("response", "No response generated.")
            mlflow.log_text(llm_response, "llm_response.txt")
            mlflow.log_param("llm_response", llm_response)
            mlflow.log_metric("llm_response_length", len(llm_response))

            mlflow.set_tag("date_time", result.get("created_at", "Unknown Time"))

    thread = threading.Thread(target=log)
    thread.daemon = True
    thread.start()
