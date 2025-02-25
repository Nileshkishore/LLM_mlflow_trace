import yaml
import mlflow
from custom_ollama import OllamaLLMWithMetadata
# Load config
from config_loader import config  


class TracedOllamaLLMWithMetadata(OllamaLLMWithMetadata):
    @mlflow.trace
    def invoke(self, prompt):
        return super().invoke(prompt)

# Initialize LLM
llm = TracedOllamaLLMWithMetadata(model=config["llm"]["model_name"])

def generate_response(prompt):
    return llm.invoke(prompt)
