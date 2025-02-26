import yaml
import mlflow
from mlflow.entities import SpanType
from custom_ollama import OllamaLLMWithMetadata
# Load config
from config_loader import config  


class TracedOllamaLLMWithMetadata(OllamaLLMWithMetadata):
    @mlflow.trace(span_type=SpanType.LLM)
    def invoke(self, prompt):
        return super().invoke(prompt)
    
    @mlflow.trace(span_type=SpanType.LLM)
    def stream(self, prompt):
        """Stream response tokens from Ollama with MLflow tracing"""
        return super().stream(prompt)

# Initialize LLM
llm = TracedOllamaLLMWithMetadata(model=config["llm"]["model_name"])

def generate_response(prompt):
    return llm.invoke(prompt)

def generate_response_stream(prompt):
    """Generate streaming response from the model"""
    return llm.stream(prompt)