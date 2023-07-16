from gpt4all import GPT4All
model = GPT4All("wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin")
output = model.generate("Once Upon A Time...", max_tokens=2048)
print(output)