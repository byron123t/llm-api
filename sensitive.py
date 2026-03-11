import os

# Single Azure key and endpoint for all models
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
azure_endpoint = os.getenv(
    "AZURE_ENDPOINT",
    "https://bjayt-mffvuqlc-eastus2.cognitiveservices.azure.com/",
)
# Optional: base URL for Mistral OCR and OpenAI-compatible chat (e.g. services.ai.azure.com)
azure_serverless_endpoint = os.getenv(
    "AZURE_SERVERLESS_ENDPOINT",
    "https://bjayt-mffvuqlc-eastus2.services.ai.azure.com",
)
