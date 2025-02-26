from langchain_ollama import OllamaLLM
import requests
import json

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
    
    def stream(self, prompt):
        """Stream tokens from Ollama API with metadata at the end"""
        url = "http://localhost:11434/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": True  # Enable streaming
        }
        
        with requests.post(url, json=data, stream=True) as response:
            response.raise_for_status()
            
            # Track metadata for final return
            metadata = {}
            streamed_response = ""  # Store streamed output
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    
                    # Collect the streamed text
                    if 'response' in chunk:
                        streamed_response += chunk['response']
                        yield chunk['response']  # Yield token-wise output
                    
                    # Save metadata when done
                    if chunk.get('done', False):
                        metadata = {
                            "model": chunk.get("model"),
                            "created_at": chunk.get("created_at"),
                            "total_duration": chunk.get("total_duration"),
                            "prompt_tokens": chunk.get("prompt_eval_count"),
                            "generated_tokens": chunk.get("eval_count")
                        }
            
            # Return full response and metadata as a dictionary after streaming completes
            return {
                "response": streamed_response,
                "metadata": metadata
            }
