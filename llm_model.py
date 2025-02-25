import yaml
import mlflow
from custom_ollama import OllamaLLMWithMetadata

# Load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

class TracedOllamaLLMWithMetadata(OllamaLLMWithMetadata):
    @mlflow.trace
    def invoke(self, prompt):
        return super().invoke(prompt)

# Initialize LLM
llm = TracedOllamaLLMWithMetadata(model=config["llm"]["model_name"])

def generate_response(prompt):
    return llm.invoke(prompt)
