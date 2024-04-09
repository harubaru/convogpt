import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "models/convogpt-3B-001"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

from transformers import pipeline

@torch.inference_mode()
def single_model():
    pipeline_1 = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    conversation_chain = [
        {'author': 'system', 'message': "Assistant is an AI language model. "},
    ]

    def render_conversation() -> str:
        rendered_str = ''
        for message in conversation_chain:
            rendered_str += f'<|im_start|>{message["author"]}\n{message["message"]}<|im_end|>\n'
        return rendered_str

    def render_conversation_readable() -> str:
        rendered_str = ''
        for message in conversation_chain:
            rendered_str += f'{message["author"]}: {message["message"]}\n'
        return rendered_str

    try:
        while True:
            conversation_chain.append({'author': 'user', 'message': input("> ")})
            prompt = render_conversation() + '<|im_start|>assistant\n'
            response = pipeline_1(prompt, max_new_tokens=256, temperature=1.0, do_sample=True, penalty_alpha=0.6, top_k=5, top_p=0.95, num_return_sequences=1, return_full_text=False)[0]["generated_text"]
            conversation_chain.append({'author': 'assistant', 'message': response})
            print(render_conversation())
    except KeyboardInterrupt:
        pass

single_model()
