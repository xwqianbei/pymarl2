modelpath=/root/autodl-tmp/Qwen2.5-7B-Instruct

# 单卡
python3 -m vllm.entrypoints.openai.api_server \
        --model $modelpath \
        --served-model-name qwen \
        --trust-remote-code \
        --tensor-parallel-size 2 \
        --port 8010

        