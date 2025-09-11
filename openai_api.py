
import os
import base64
from openai import AzureOpenAI
from sensitive import azure_key_1, azure_key_2, azure_location, azure_endpoint
from image_encode import local_image_to_data_url


class OpenAIAPI:
    
    def __init__(self, deployment):
        self.endpoint = azure_endpoint
        self.subscription_key = azure_key_1
        self.deployment = deployment
        if deployment != "gpt-4o" and deployment != "o4-mini" and deployment != "gpt-4.1-mini":
            raise ValueError("Invalid deployment name. Must be 'gpt-4o', 'o4-mini', or 'gpt-4.1-mini'.")
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
            api_version="2025-01-01-preview",
        )
    
    def chat_completion(self, messages, max_tokens=16384):
        completion = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_completion_tokens=max_tokens,
            stop=None,
            stream=False
        )
        return completion

