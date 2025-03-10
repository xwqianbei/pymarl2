import json
import mmap
import os
import sys

def mmap_read(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        # 创建内存映射文件对象
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            start = 0
            while True:
                # 查找下一个换行符
                end = mm.find(b'\n', start)
                if end == -1:  # 处理最后一行
                    line = mm[start:]
                    if not line:
                        break
                else:
                    line = mm[start:end]

                # 解析 JSON 行
                if line:
                    try:
                        json_obj = json.loads(line.decode("utf-8"))
                        # 在此处理 JSON 对象（示例：打印）
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"JSON 解析失败: {e}, 行内容: {line}")

                # 移动到下一行起始位置
                start = end + 1 if end != -1 else len(mm)
    return data

def save_jsonl(file_path, data):
    assert isinstance(data, list), "save_jsonl: data should be a list."
    with open(file_path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

def save_json(file_path, data):
    assert isinstance(data, dict), "save_json: data should be a dict."
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))