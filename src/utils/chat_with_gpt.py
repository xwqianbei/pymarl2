import requests
import os
import json
from utils.prompt_template import *
import numpy as np
import argparse
import openai


def gpt_json_load(json_str):
    # 去除代码块标记和换行符
    json_str = json_str.replace("```json", "").replace("```", "").replace("\n", "").strip()
    # 尝试解析JSON
    try:
        json_data = json.loads(json_str)
    except json.JSONDecodeError:
        # 如果JSON解析失败，尝试将其作为Python字面量解析
        import ast
        python_data = ast.literal_eval(json_str)
        # 将Python数据结构转换为JSON兼容的格式
        json_data = json.loads(json.dumps(python_data))
    return json_data

def save_gpt_response_detail(states, gpt_response, refine_response):
    content = {
        'states': str(states),
        'gpt_response': gpt_response,
        'refine_response': refine_response
    }
    with open('data/response_detail.json', 'a', encoding='utf-8') as f:
        f.write(json.dumps(content, ensure_ascii=False) + '\n')

def call_openai(model, messages, temperature=0.7, response_format=None):
    openai.api_key = os.getenv("ONE_API_KEY")
    openai.api_base = "https://one-api.glm.ai/v1"
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
    )
    return response.choices[0].message.content

"""
curl http://localhost:8010/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"}
        ]
    }
'"""
def call_qwen(messages, model='qwen', response_format=None):
    url = "http://localhost:8010/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "response_format": response_format
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']


def callgpt(env_name, map_name, env_states, num_agent):
    if 'sc2' in env_name:
        prompt = SC2_prompt(map_name)
    
    # save_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + f'/response/{env_name}/{map_name}/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    
    model = 'gpt-4o-mini-2024-07-18'
    temperature = 0.7
    
    message = prompt.get_message(env_states)
    # gpt_response = call_openai(model, message, temperature, response_format={"type": "json_object"})
    gpt_response = call_qwen(message, model='qwen', response_format={"type": "json_object"})
    refine_response = parse_response(gpt_response, num_agent)
    save_gpt_response_detail(env_states, gpt_response, refine_response)    
    return refine_response
    

def parse_response(response, num_agent):
    try:
        res = gpt_json_load(response)
        n_agent_res = []
        for id, agent_res in enumerate(res, start=0):
            try:
                assert 'agentID' in agent_res and "skill" in agent_res and 'conditions' in agent_res and 'thought_process' in agent_res
                assert agent_res['skill'] in ['Focus Fire', 'Retreat',  'Spread Out', 'Advance']
                if 'agentID' in agent_res and isinstance(agent_res['agentID'], str):
                    agent_res['agentID'] = int(agent_res['agentID'])
                assert agent_res['agentID'] == id
                n_agent_res.append(agent_res)
            except:
                default_response = {
                    "agentID": id,
                    "skill": "Focus Fire",
                    "conditions": "default assignment",
                    "thought_process": "default assignment"
                }
                n_agent_res.append(default_response)
        assert len(n_agent_res) == num_agent
        return n_agent_res
    except:
        default_n_agent_res = [
            {
                "agentID": id,
                "skill": "Focus Fire",
                "conditions": "default assignment",
                "thought_process": "default assignment"
            } for id in range(num_agent)
        ]
        return default_n_agent_res
    

    