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
        """Stream tokens and return full response with metadata"""
        url = "http://localhost:11434/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }

        full_response = ""  # Store full response text
        metadata = {}  # Store metadata

        with requests.post(url, json=data, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))

                    if 'response' in chunk:
                        token = chunk['response']
                        full_response += token  # Append token to full response
                        yield token  # Still allows streaming

                    if chunk.get('done', False):
                        metadata = {
                            "model": chunk.get("model"),
                            "created_at": chunk.get("created_at"),
                            "total_duration": chunk.get("total_duration"),
                            "prompt_tokens": chunk.get("prompt_eval_count"),
                            "generated_tokens": chunk.get("eval_count"),
                            "full_response": full_response  # Final full response
                        }

            yield metadata  # Yield metadata at the end
