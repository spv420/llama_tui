from llama_cpp import Llama
import json

llm = Llama(model_path="/home/spv/llama.cpp/ggml-vicuna13b-q2_k.bin", verbose=False)

output = llm("### Human: Hi there, hello\n### Assistant:", max_tokens=32, stop=["### Human:"], echo=False, stream=True)

for tok in output:
	print(tok["choices"][0]["text"], flush=True)
