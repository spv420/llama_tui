from llama_cpp import Llama
from rich import print
import json
import sys

llm = Llama(model_path="/home/spv/llama_tui/wizardvicuna-7b-q2_K.bin", verbose=False, n_threads=4)

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
	m = ""
	print("@", end="", flush=True)
	for tok in send_message(s):
		history += tok
		m += tok
		print(tok, end="", flush=True)
#	print(m, flush=True)
	print()
