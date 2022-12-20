from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("models/convogpt-small/hakurei/lit-125M-10-1-0.0001--1671505121")

def generate(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
    output = model.generate(input_ids=input_ids, max_length=100, do_sample=True, top_p=0.95, top_k=60)
    return tokenizer.decode(output[0])

print(generate("haru: hi\nMaribel Hearn:"))