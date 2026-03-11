# TL;DR
This repo contains examples for running inferences, annotations, and labels on images using VLM and related capabilities (OpenAI Azure, Mistral Document AI, image generation). It supports multimodal inputs and image encoding.

# Version
Use Python 3.11.13 (or 3.8+). Recommend keeping a conda env or venv and using pyenv to install the correct version of Python locally.

# Libraries
```bash
pip install -r requirements.txt
```

# Environment variables
All models use a **single** Azure API key.

- **`AZURE_API_KEY`** (required) — Used for every model (chat, Document AI, image gen/edit).
- **`AZURE_ENDPOINT`** (optional) — Azure OpenAI base URL for chat and image generation. Default: `https://bjayt-mffvuqlc-eastus2.cognitiveservices.azure.com/`
- **`AZURE_SERVERLESS_ENDPOINT`** (optional) — Base URL for Mistral Document AI and OpenAI-compatible chat (e.g. DeepSeek, Grok). Default: `https://bjayt-mffvuqlc-eastus2.services.ai.azure.com`

Example:
```bash
export AZURE_API_KEY="<your-api-key>"
# optional:
export AZURE_ENDPOINT="https://your-cognitiveservices.azure.com/"
export AZURE_SERVERLESS_ENDPOINT="https://your-services.ai.azure.com"
```

# Supported models

- **Chat (Azure):** `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini`, `gpt-5`, `gpt-5-chat`, `gpt-5.1`, `gpt-5.2`, `gpt-5.2-chat`, `gpt-5.3-codex`, `gpt-5.4`, `gpt-5.4-pro`, `o3`, `o3-pro`, `o4-mini`
- **Chat (OpenAI-compatible):** `DeepSeek-V3.2`, `grok-4-1-fast-non-reasoning`, `grok-4-1-fast-reasoning`
- **Document AI:** `mistral-document-ai-2512` (OCR; base64 input only)
- **Image:** `gpt-image-1.5` (generation and edit)

# API usage

## Chat (all chat models)
Same interface for Azure and OpenAI-compatible chat models. Supports multimodal messages (text + image).

```python
from openai_api import OpenAIAPI

client = OpenAIAPI(deployment="gpt-4o")  # or "DeepSeek-V3.2", "grok-4-1-fast-reasoning", etc.
output = client.chat_completion(messages, max_tokens=16384)
print(output.choices[0].message.content)
```

## Prompting and multimodal (images in chat)
Edit prompts in `prompts.json` (same role/content format). To attach an image to a prompt:

```python
from prompts import PROMPTS
from image_encode import local_image_to_data_url

# Initialize the image block and set its URL
image = PROMPTS["Image"].copy()
image["image_url"] = {"url": local_image_to_data_url("image_path.png")}

# Append to your base prompt
base_prompt = PROMPTS["CaptionSimple"].copy()
base_prompt[1]["content"] = base_prompt[1]["content"] + [image]

# Call chat
from openai_api import OpenAIAPI
client = OpenAIAPI(deployment="gpt-4o")
output = client.chat_completion(base_prompt, max_tokens=16384)
```

Chat models support `image_url` with base64 data URLs (from `local_image_to_data_url`).

## Mistral Document AI (mistral-document-ai-2512)
Document AI uses a dedicated OCR endpoint. **Only base64 input is supported** (document or image URL is not supported).

Build the document payload with base64 data URLs:

- **PDF:** use `local_file_to_data_url("file.pdf")` → `data:application/pdf;base64,...`
- **Image:** use `local_image_to_data_url("image.jpg")` → `data:image/jpeg;base64,...`

Example (PDF):
```python
from image_encode import local_file_to_data_url
from openai_api import document_ocr

pdf_data_url = local_file_to_data_url("document.pdf")
document_payload = {"type": "document_url", "document_url": pdf_data_url}
result = document_ocr(document_payload, include_image_base64=True)
```

Example (image):
```python
from image_encode import local_image_to_data_url
from openai_api import document_ocr

image_data_url = local_image_to_data_url("page.png")
document_payload = {"type": "image_url", "image_url": image_data_url}
result = document_ocr(document_payload, include_image_base64=True)
```

Optional: pass `bbox_annotation_format` or `document_annotation_format` (JSON schema dicts) for structured output.

## Image generation and edit (gpt-image-1.5)

**Generate an image:**
```python
from openai_api import image_generation
import base64

resp = image_generation(
    prompt="A photograph of a red fox in an autumn forest",
    size="1024x1024",
    quality="medium",
    output_format="png",
    n=1,
)
# Decode and save
b64 = resp["data"][0]["b64_json"]
with open("generated_image.png", "wb") as f:
    f.write(base64.b64decode(b64))
```

**Edit an image** (with mask):
```python
from openai_api import image_edit
import base64

resp = image_edit(
    image_path="image_to_edit.png",
    mask_path="mask.png",
    prompt="Make this black and white",
)
b64 = resp["data"][0]["b64_json"]
with open("edited_image.png", "wb") as f:
    f.write(base64.b64decode(b64))
```

# Testing
Run from the project root:
```bash
python caption.py
```
This uses `gpt-4o` and a test image; ensure `AZURE_API_KEY` is set.

# API key
Email bjaytang@umich.edu for the Azure key. Then:
```bash
export AZURE_API_KEY="<key>"
```
or with conda:
```bash
conda env config vars set AZURE_API_KEY="<key>"
```
