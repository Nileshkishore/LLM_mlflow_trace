from langchain_ollama import OllamaLLM
import requests
import json
import mlflow
from mlflow.entities import SpanType

class OllamaLLMWithMetadata(OllamaLLM):
    def invoke(self, prompt):
        """Override invoke method to include metadata"""
        url = "http://localhost:11434/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False  # Ensure we get full metadata in one response
        }
        
        response = requests.post(url, json=data)
        response_json = response.json()  # Parse JSON
        
        return {
            "model": response_json.get("model"),
            "response": response_json.get("response"),
            "created_at": response_json.get("created_at"),
            "total_duration": response_json.get("total_duration"),
            "prompt_tokens": response_json.get("prompt_eval_count"),
            "generated_tokens": response_json.get("eval_count")
        }
    
    @mlflow.trace(span_type=SpanType.LLM)
    def get_metadata(full_response, metadata):
        return full_response, metadata
    
    def stream(self, prompt):
        """Stream tokens from Ollama API and return the full response."""
        url = "http://localhost:11434/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": True  # Enable streaming
        }

        full_response = ""
        metadata = {}

        with requests.post(url, json=data, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))

                    # Collect response tokens
                    if 'response' in chunk:
                        full_response += chunk['response']
                        yield chunk['response']  # Still stream it in real-time

                    # Store metadata when streaming is complete
                    if chunk.get('done', False):
                        metadata = {
                            "model": chunk.get("model"),
                            "created_at": chunk.get("created_at"),
                            "total_duration": chunk.get("total_duration"),
                            "prompt_tokens": chunk.get("prompt_eval_count"),
                            "generated_tokens": chunk.get("eval_count"),
                        }
       
        self.get_metadata(self,full_response, metadata)