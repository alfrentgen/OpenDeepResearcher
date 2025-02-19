#!/bin/bash

#model_path="./models/llama3.3/Llama-3.3-70B-Instruct-Q4_K_L.gguf"
model_path="./models/deepseek/DeepSeek-R1-Distill-Qwen-32B-Q4_K_L.gguf" #36: 8,22,22
#model_path="./models/deepseek/DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf"  #48: 8,22,22

host="HOST_IP"
port="8080"
layers="48" #"32"
split="8,22,22" #"1,5,5"
ctx_size="98304" #"65536" #"65536"
parseq="2"

./llama_bins/llama-server --model ${model_path} \
--port ${port} --host ${host} \
-ngl ${layers} \
--ctx-size ${ctx_size} \
-ts ${split} \
-fa -nkvo \
--log-colors --verbose \
-cb -np ${parseq} \
2>server_err.log 1>server_out.log&
