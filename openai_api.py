import os
import requests
from openai import AzureOpenAI, OpenAI
from sensitive import (
    AZURE_API_KEY,
    azure_endpoint,
    azure_serverless_endpoint,
)

# Model groups
AZURE_CHAT_MODELS = {
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5-chat",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.2-chat",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-pro",
    "o3",
    "o3-pro",
    "o4-mini",
}
OPENAI_COMPATIBLE_CHAT_MODELS = {
    "DeepSeek-V3.2",
    "grok-4-1-fast-non-reasoning",
    "grok-4-1-fast-reasoning",
}
MISTRAL_OCR_MODEL = "mistral-document-ai-2512"
IMAGE_GEN_MODEL = "gpt-image-1.5"

ALL_CHAT_MODELS = AZURE_CHAT_MODELS | OPENAI_COMPATIBLE_CHAT_MODELS
AZURE_API_VERSION = "2024-12-01-preview"
IMAGE_API_VERSION = "2024-02-01"


class OpenAIAPI:
    def __init__(self, deployment):
        self.deployment = deployment
        if deployment not in ALL_CHAT_MODELS:
            raise ValueError(
                f"Invalid deployment name. Must be one of {sorted(ALL_CHAT_MODELS)}."
            )
        self._api_key = AZURE_API_KEY
        if deployment in AZURE_CHAT_MODELS:
            self._client = AzureOpenAI(
                azure_endpoint=azure_endpoint.rstrip("/") + "/",
                api_key=self._api_key,
                api_version=AZURE_API_VERSION,
            )
        else:
            base_url = azure_serverless_endpoint.rstrip("/") + "/openai/v1"
            self._client = OpenAI(
                base_url=base_url,
                api_key=self._api_key,
            )

    def chat_completion(self, messages, max_tokens=16384):
        completion = self._client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_completion_tokens=max_tokens,
            stop=None,
            stream=False,
        )
        return completion


def document_ocr(
    document_payload,
    bbox_annotation_format=None,
    document_annotation_format=None,
    include_image_base64=True,
):
    """Call Mistral Document AI (mistral-document-ai-2512). Only base64 document/image
    data URLs are supported; raw URLs are not supported.
    document_payload: dict with e.g. {"type": "document_url", "document_url": "data:application/pdf;base64,..."}
    or {"type": "image_url", "image_url": "data:image/jpeg;base64,..."}
    """
    url = f"{azure_serverless_endpoint.rstrip('/')}/providers/mistral/azure/ocr"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_API_KEY}",
    }
    body = {
        "model": MISTRAL_OCR_MODEL,
        "document": document_payload,
        "include_image_base64": include_image_base64,
    }
    if bbox_annotation_format is not None:
        body["bbox_annotation_format"] = bbox_annotation_format
    if document_annotation_format is not None:
        body["document_annotation_format"] = document_annotation_format
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()


def image_generation(
    prompt,
    size="1024x1024",
    quality="medium",
    output_compression=100,
    output_format="png",
    n=1,
):
    """Generate an image with gpt-image-1.5. Returns the raw API response; use
    response['data'][0]['b64_json'] for the base64 image, or response['data'][0]['url'] if present.
    """
    base = azure_endpoint.rstrip("/")
    url = f"{base}/openai/deployments/{IMAGE_GEN_MODEL}/images/generations?api-version={IMAGE_API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_API_KEY}",
    }
    body = {
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "output_compression": output_compression,
        "output_format": output_format,
        "n": n,
    }
    resp = requests.post(url, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()


def image_edit(image_path, mask_path, prompt):
    """Edit an image with gpt-image-1.5 using a mask. Returns the raw API response;
    use response['data'][0]['b64_json'] for the base64 image.
    """
    base = azure_endpoint.rstrip("/")
    url = f"{base}/openai/deployments/{IMAGE_GEN_MODEL}/images/edits?api-version={IMAGE_API_VERSION}"
    headers = {"Authorization": f"Bearer {AZURE_API_KEY}"}
    with open(image_path, "rb") as img, open(mask_path, "rb") as msk:
        files = {
            "image": (os.path.basename(image_path), img, "application/octet-stream"),
            "mask": (os.path.basename(mask_path), msk, "application/octet-stream"),
        }
        data = {"prompt": prompt}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
    resp.raise_for_status()
    return resp.json()
