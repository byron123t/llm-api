# TL;DR
This repo contains some examples for outputing inferences/annotations/labels on images using the VLM capabilities of OpenAI Azure API.
# Version
Use Python 3.11.13
Recommend keeping a conda env or venv and using pyenv to install the correct version of Python and using it locally.
# Libraries
Use `pip install -r requirements.txt`
# Testing
Run `python caption.py` from within `api/` to test if it recognizes the doge. Don't forget to update your API key environment variable.
# Prompting
Edit prompts in the `prompts.json` file, maintaining the same role, content format.
Use the following to initialize the image:
```python
from prompts import PROMPTS
image = PROMPTS['Image']
```
Get the image encoded in the proper format with the following:
```python
from image_encode import local_image_to_data_url
imageurl = local_image_to_data_url('image_path.png')
```
Edit the input image url from the initialized image:
```python
image['image_url']['url'] = imageurl
```
Append the new updated image with the encoded image url to the base prompt you defined in `prompts.json`
```python
from prompts import PROMPTS
base_prompt = PROMPTS['CaptionSimple']
base_prompt[1]['content'].append(image)
```
# API Usage
```python
from openai_api import OpenAIAPI
client = OpenAIAPI(deployment=model)
output = client.chat_completion(base_prompt, max_tokens=max_tokens)
```
# API Key
Email bjaytang@umich.edu for the Azure keys.
Run:
```bash
conda env config vars set AZURE_API_KEY_1="key_from_brian" AZURE_API_KEY_2="key_2_from_brian"
```
or
```bash
export AZURE_API_KEY_1="key_from_brian"
export AZURE_API_KEY_2="key_2_from_brian"
```