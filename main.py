from llama_cpp import Llama
import json
import sys

llm = Llama(model_path="/home/spv/llama.cpp/ggml-vicuna13b-q2_k.bin", verbose=False, n_threads=4)

history = ""

def send_message(s):
	global history
	history += "### Human: " + s + "\n### Assistant:"

	output = llm(history, max_tokens=256, stop=["### Human:"], echo=False, stream=True)

	for tok in output:
		yield tok["choices"][0]["text"]

	history += "\n"

while True:
	s = input("> ")
	print("@", end="", flush=True)
	for tok in send_message(s):
		history += tok
		print(tok, end="", flush=True)
	print()
