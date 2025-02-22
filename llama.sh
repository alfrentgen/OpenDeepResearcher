#!/bin/bash

#model_path="./models/llama3.3/Llama-3.3-70B-Instruct-Q4_K_L.gguf"
#model_path="./models/deepseek/DeepSeek-R1-Distill-Qwen-32B-Q4_K_L.gguf" #61: 130,435,435: 131072,q8_0,q8_0
model_path="./models/deepseek/DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf"  #48: 8,22,22

host="HOST_IP"
port="8080"
layers="60" #"32"
split="110,445,445" #"8,22,22" #"1,5,5"
ctx_size="131072" #"98304" #"106496" #"65536" #"131072"
parseq="2"
ubatch="1048"

# draft parameters
./llama_bins/llama-server --model ${model_path} \
--port ${port} --host ${host} \
-ngl ${layers} \
--ctx-size ${ctx_size} -ctk q8_0 -ctv q8_0 \
-ub ${ubatch} \
-ts ${split} \
-fa -nkvo \
--log-colors --verbose \
-cb -np ${parseq} \
--no-warmup \
--prio 3 --threads 8 \
--log-file llama-server.log --log-timestamps --log-verbose \
2>server_err.log 1>server_out.log&

