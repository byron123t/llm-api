import os
import requests
from openai import AzureOpenAI, OpenAI
from sensitive import (
    AZURE_API_KEY,
    azure_endpoint,
    azure_serverless_endpoint,
    azure_responses_api_version,
)

# Azure models that must use the Responses API (/openai/v1/responses), not chat.completions.
# See: https://learn.microsoft.com/azure/ai-services/openai/how-to/responses
AZURE_RESPONSES_API_MODELS = frozenset(
    {"gpt-5.3-codex", "gpt-5.4-pro", "o3-pro"}
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
# Used only for AzureOpenAI chat.completions — not for /openai/v1/responses.
AZURE_API_VERSION = "2024-12-01-preview"

# If the preferred Responses api-version is rejected, try these in order (after the env default).
_RESPONSES_API_VERSION_FALLBACKS = (
    "preview",
    "2025-03-01-preview",
    "2025-04-01-preview",
)


class _ChatCompletionShim:
    """Minimal object so `output.choices[0].message.content` works for Responses API."""

    def __init__(self, content: str):
        self.choices = [_ChoiceShim(content)]


class _ChoiceShim:
    def __init__(self, content: str):
        self.message = _MessageShim(content)


class _MessageShim:
    def __init__(self, content: str):
        self.content = content


def _messages_to_responses_payload(messages):
    """Convert chat-style messages to Azure Responses API `instructions` + `input`."""
    instruction_parts = []
    input_items = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str):
                instruction_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        instruction_parts.append(part.get("text", ""))
            continue
        if role not in ("user", "assistant"):
            continue
        if isinstance(content, str):
            input_items.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                parts.append({"type": "input_text", "text": part.get("text", "")})
            elif ptype == "image_url":
                iu = part.get("image_url") or {}
                url = iu.get("url") if isinstance(iu, dict) else iu
                if url:
                    parts.append({"type": "input_image", "image_url": url})
        if parts:
            input_items.append({"role": role, "content": parts})
    instructions = "\n\n".join(p for p in instruction_parts if p.strip()) or None
    return instructions, input_items


def _extract_text_from_responses_body(data: dict) -> str:
    """Pull assistant text from a Responses API JSON body."""
    texts = []
    for item in data.get("output") or []:
        if item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            btype = block.get("type")
            if btype == "output_text" and block.get("text"):
                texts.append(block["text"])
            elif btype in ("text", "input_text") and block.get("text"):
                texts.append(block["text"])
    if texts:
        return "\n".join(texts)
    # Some payloads expose a single string
    if data.get("output_text"):
        return str(data["output_text"])
    return ""


def _responses_api_chat(deployment: str, messages, max_tokens: int) -> _ChatCompletionShim:
    base = azure_endpoint.rstrip("/")
    url = f"{base}/openai/v1/responses"
    instructions, input_payload = _messages_to_responses_payload(messages)
    if not input_payload:
        raise ValueError("No user/assistant messages to send to the Responses API.")
    body = {
        "model": deployment,
        "input": input_payload,
        "max_output_tokens": max_tokens,
    }
    if instructions:
        body["instructions"] = instructions
    headers = {
        "Content-Type": "application/json",
        "Api-Key": AZURE_API_KEY,
    }
    # Build version list: user/env first, then fallbacks (deduped).
    versions = [azure_responses_api_version]
    for v in _RESPONSES_API_VERSION_FALLBACKS:
        if v not in versions:
            versions.append(v)

    last_detail = None
    resp = None
    for api_ver in versions:
        resp = requests.post(
            url,
            headers=headers,
            params={"api-version": api_ver},
            json=body,
            timeout=600,
        )
        if resp.ok:
            break
        try:
            detail = resp.json()
            last_detail = detail
            err_msg = str((detail.get("error") or {}).get("message", "")).lower()
        except Exception:
            last_detail = resp.text
            err_msg = ""
        is_version_reject = resp.status_code == 400 and (
            "api version" in err_msg and "not supported" in err_msg
        )
        if is_version_reject and api_ver != versions[-1]:
            continue
        raise RuntimeError(
            f"Responses API {resp.status_code} {resp.reason} (api-version={api_ver}): {last_detail}"
        )

    data = resp.json()
    if data.get("error"):
        raise RuntimeError(data["error"])
    text = _extract_text_from_responses_body(data)
    return _ChatCompletionShim(text)
# Match Azure Images reference for gpt-image-1.5
IMAGE_API_VERSION = "2025-04-01-preview"


class OpenAIAPI:
    def __init__(self, deployment):
        self.deployment = deployment
        if deployment not in ALL_CHAT_MODELS:
            raise ValueError(
                f"Invalid deployment name. Must be one of {sorted(ALL_CHAT_MODELS)}."
            )
        self._api_key = AZURE_API_KEY
        self._use_azure_responses = deployment in AZURE_RESPONSES_API_MODELS
        if self._use_azure_responses:
            self._client = None
        elif deployment in AZURE_CHAT_MODELS:
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
        if self._use_azure_responses:
            return _responses_api_chat(self.deployment, messages, max_tokens)
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
    # Use `Api-Key` header as in Azure REST examples
    headers = {
        "Content-Type": "application/json",
        "Api-Key": AZURE_API_KEY,
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


def image_edit(
    image_path,
    mask_path,
    prompt,
    size="1024x1024",
    quality="medium",
    output_compression=100,
    output_format="png",
    n=1,
):
    """Edit an image with gpt-image-1.5 using a mask. Returns the raw API
    response; use response['data'][0]['b64_json'] for the base64 image.
    Same contract as image_edit, but includes a mask file.
    """
    base = azure_endpoint.rstrip("/")
    url = f"{base}/openai/deployments/{IMAGE_GEN_MODEL}/images/edits?api-version={IMAGE_API_VERSION}"
    # Use `Api-Key` header, no explicit Content-Type (requests will set multipart boundary)
    headers = {"Api-Key": AZURE_API_KEY}

    # Follow Azure Images Edit reference: multipart/form-data with image (+ optional mask)
    # and form fields: prompt, n, size, quality.
    with open(image_path, "rb") as img, open(mask_path, "rb") as msk:
        files = {
            "image": (os.path.basename(image_path), img, "image/jpeg"),
            "mask": (os.path.basename(mask_path), msk, "image/png"),
        }
        data = {
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": quality,
        }
        resp = requests.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()
