import openai
import json
import requests




class Call_API(object):
    def __init__(self, args):
        self.args = args

    def call_openai(self, messages, response_format=None):
        openai.api_key = "sk-7I4034OzkVENe2flOxwRZo4bZ27IQu4MMXxObBaRtQVDn1c8"
        openai.api_base = "https://api2.aigcbest.top/v1"
        response = openai.ChatCompletion.create(
            model=self.args.model,
            messages=messages,
            response_format=response_format,
        )
        return response.choices[0].message.content


    def call_deepseek(self, messages, response_format=None):
        if self.args.deepseek_platform == 'deepseek':
            openai.api_key = ""
            openai.api_base = ""
            response = openai.ChatCompletion.create(
                model=self.args.model,
                messages=messages,
                stream=False,
                response_format=response_format,
            )
            return response.choices[0].message.content
        elif self.args.deepseek_platform == 'tencent':
            url = "http://localhost:8434/api/generate"
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "model": "deepseek-r1:32b",
                "prompt": messages,
                "stream": False
            }
            response = requests.post(url, headers=headers, json=data)
            return response.json()['choices'][0]['message']['content']
        elif self.args.deepseek_platform == 'gjld':
            url = "https://api.siliconflow.cn/v1/chat/completions"
            payload = {
                "model": "deepseek-ai/DeepSeek-R1",
                "messages": messages,
                "stream": False,
                "max_tokens": 512,
                "stop": ["null"],
                "temperature": 0.7,
                "top_p": 0.7,
                "top_k": 50,
                "frequency_penalty": 0.5,
                "n": 1,
                "response_format":  {"type": "text"},
            }
            headers = {
                "Authorization": "Bearer sk-pjblsqzuikqzimgzokodqudxnoqbsbmotgvuaumrabkpunmp",
                "Content-Type": "application/json"
            }
            response = requests.request("POST", url, json=payload, headers=headers)
            return response.text
        else:
            raise ValueError(f"Deepseek platform {self.args.deepseek_platform} not supported")
            return None
        

    def call_qwen(self, messages, response_format=None):
        openai.api_key = 'sk-c1b899f53dca4e36a6a05ec6c44357f8'
        openai.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1" 
        response = openai.ChatCompletion.create(
            model = self.args.model, # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages = messages,
        )
            
        return response.choices[0].message.content

    def gpt_json_load(self, json_str):
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

    def parse_response(self, response, num_agent):
        try:
            res = self.gpt_json_load(response)

            save_path = 'src/dataset/2s3z_model_output.jsonl'
            with open(save_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

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
        except Exception as e:
            default_n_agent_res = [
                {
                    "agentID": id,
                    "skill": "Focus Fire",
                    "conditions": "default assignment",
                    "thought_process": "default assignment"
                } for id in range(num_agent)
            ]
            print(f"Error: {e}")
            import traceback
            print(traceback.print_exc())
            return default_n_agent_res

    def __call__(self, messages, num_agent, response_format=None):
        try: 
            if self.args.api_type == 'openai':
                response = self.call_openai(messages, response_format)
            elif self.args.api_type == 'deepseek':
                response = self.call_deepseek(messages, response_format)
            elif self.args.api_type == 'qwen':
                response = self.call_qwen(messages, response_format)
            else:
                raise ValueError(f"Model {self.args.api_type} not supported")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            print(traceback.print_exc())
        
        response = self.parse_response(response, num_agent)
        return response
    


    