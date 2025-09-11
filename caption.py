import os
from openai_api import OpenAIAPI
from prompts import PROMPTS
from image_encode import local_image_to_data_url


model = 'gpt-4o'
client = OpenAIAPI(deployment=model)

base_prompt = PROMPTS['CaptionSimple']
image = 'static/test.png'
imageurl = local_image_to_data_url(image)
image = PROMPTS['Image']
image['image_url']['url'] = imageurl
base_prompt[1]['content'].append(image)
output = client.chat_completion(base_prompt, max_tokens=16384)
print('========== Full Output ==========')
print(output)
print('=================================')
print('========== Simple Caption ==========')
print(output.choices[0].message.content)
print('=================================')

base_prompt = PROMPTS['CaptionDetailed']
image = 'static/test.png'
imageurl = local_image_to_data_url(image)
image = PROMPTS['Image']
image['image_url']['url'] = imageurl
base_prompt[1]['content'].append(image)
output = client.chat_completion(base_prompt, max_tokens=16384)
print('========== Full Output ==========')
print(output)
print('=================================')
print('========== Detailed Caption ==========')
print(output.choices[0].message.content)
print('=================================')
