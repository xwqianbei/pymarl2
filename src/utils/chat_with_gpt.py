import requests
import os
import json
from utils.prompt_template import *
import numpy as np
import argparse
from openai import OpenAI

def inference(model, message, temperature, host):
    data=json.dumps({'model':model,'messages':message, 'temperature':temperature,'response_format':{ "type": "json_object" }})
    if host is not None:
        out = requests.post(host,data=data)
        out_content=json.loads(out.text)['choices'][0]['message']['content']
    else:
        client = OpenAI()
        out_content = client.chat.completions.create(model=model, messages=message, temperature=temperature, response_format='json_object')


    return out_content

def callgpt(args, env_name, map_name, save=False, id=0, use_recheck=True):
    if 'sc2' in env_name:
        prompt = SC2_prompt(map_name)
    
    message = prompt.get_message()
    save_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + f'/response/{env_name}/{map_name}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model = 'gpt-4o-mini-2024-07-18'
    temperature = 0.0
    host = None
    out_content = inference(model, message, temperature, host)
    print(out_content)
    

    # TODO: check the response and return the agents' roles
    