import copy
import base64
import os
from openai_api import (
    OpenAIAPI,
    document_ocr,
    image_generation,
    IMAGE_GEN_MODEL,
)
from prompts import PROMPTS
from image_encode import local_image_to_data_url, local_file_to_data_url


def build_caption_messages(prompt_name, image_path):
    """Build a fresh message list with the image attached. Uses deepcopy so
    PROMPTS is never mutated when iterating over multiple samples.
    """
    base_prompt = copy.deepcopy(PROMPTS[prompt_name])
    image_block = copy.deepcopy(PROMPTS["Image"])
    image_block["image_url"] = {"url": local_image_to_data_url(image_path)}
    base_prompt[1]["content"] = base_prompt[1]["content"] + [image_block]
    return base_prompt


def run_chat_demo(model_name, messages, max_tokens=16384):
    """Run chat completion and print only the model output."""
    client = OpenAIAPI(deployment=model_name)
    output = client.chat_completion(messages, max_tokens=max_tokens)
    content = output.choices[0].message.content
    print(f"{model_name}:")
    print(content)
    print()
    return content


def run_image_generation_demo(save_path="static/generated_image.png"):
    """Run gpt-image-1.5 generation demo; print only the save path."""
    resp = image_generation(
        prompt="A photograph of a red fox in an autumn forest",
        size="1024x1024",
        quality="medium",
        output_format="png",
        n=1,
    )
    b64 = resp["data"][0]["b64_json"]
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"{IMAGE_GEN_MODEL}: saved to {os.path.abspath(save_path)}")
    print()
    return save_path


# Test image used for caption demos
IMAGE_PATH = "static/test.png"

# Chat models that support vision/image inputs (used for caption demos)
CHAT_VISION_DEMO_MODELS = [
    "gpt-4o",
    "gpt-5.4",
    "grok-4-1-fast-reasoning",
]

# Text-only model (no vision); demo with a non-image prompt
DEEPSEEK_MODEL = "DeepSeek-V3.2"

# Paths for Mistral Document AI OCR demos (base64 only)
OCR_IMAGE_PATH = "static/test_ocr.png"
OCR_PDF_PATH = "static/test_ocr.pdf"

def print_mistral_ocr_output(label, result):
    """Print only the extracted markdown from Mistral OCR result (no raw dict)."""
    pages = result.get("pages") or []
    for i, page in enumerate(pages):
        markdown = page.get("markdown") or ""
        if markdown.strip():
            print(f"{label}:")
            print(markdown.strip())
            print()


if __name__ == "__main__":
    # Simple caption with vision-capable models only
    messages_simple = build_caption_messages("CaptionSimple", IMAGE_PATH)
    for model_name in CHAT_VISION_DEMO_MODELS:
        try:
            run_chat_demo(model_name, messages_simple)
        except Exception as e:
            print(f"{model_name}: Error — {e}\n")

    # Detailed caption (gpt-4o)
    messages_detailed = build_caption_messages("CaptionDetailed", IMAGE_PATH)
    try:
        run_chat_demo("gpt-4o", messages_detailed)
    except Exception as e:
        print(f"gpt-4o: Error — {e}\n")

    # DeepSeek-V3.2: text-only
    text_messages = [{"role": "user", "content": "What is the capital of France? Reply in one sentence."}]
    try:
        run_chat_demo(DEEPSEEK_MODEL, text_messages)
    except Exception as e:
        print(f"{DEEPSEEK_MODEL}: Error — {e}\n")

    # Image generation
    try:
        run_image_generation_demo()
    except Exception as e:
        print(f"{IMAGE_GEN_MODEL}: Error — {e}\n")

    # Mistral Document AI OCR: print only extracted markdown
    if os.path.isfile(OCR_IMAGE_PATH):
        try:
            doc = {"type": "image_url", "image_url": local_image_to_data_url(OCR_IMAGE_PATH)}
            result = document_ocr(doc, include_image_base64=True)
            print_mistral_ocr_output("mistral-document-ai-2512 (image OCR)", result)
        except Exception as e:
            print(f"mistral-document-ai-2512 (image OCR): Error — {e}\n")
    else:
        print(f"Skipping image OCR: {OCR_IMAGE_PATH} not found.\n")

    if os.path.isfile(OCR_PDF_PATH):
        try:
            doc = {"type": "document_url", "document_url": local_file_to_data_url(OCR_PDF_PATH)}
            result = document_ocr(doc, include_image_base64=True)
            print_mistral_ocr_output("mistral-document-ai-2512 (PDF OCR)", result)
        except Exception as e:
            print(f"mistral-document-ai-2512 (PDF OCR): Error — {e}\n")
    else:
        print(f"Skipping PDF OCR: {OCR_PDF_PATH} not found.\n")
