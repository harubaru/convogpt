from transformers import AutoTokenizer, AutoModelForCausalLM

class GPTGenerator:
    def __init__(self, model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path
        ).eval()

    def generate(self, prompt: str, max_length: int, count: int):
        output = {}

        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors='pt'
        ).repeat(count, 1)

        output = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            max_new_tokens=max_length,
            temperature=0.5,
            top_p=0.9,
            typical_p=0.98,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=198,
        )[:, input_ids.shape[1]:]

        output = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True
        )

        return output

generator = GPTGenerator("models/convogpt-small/hakurei/lit-125M-1-2-1e-05--1671518439")
print(generator.generate("Eiki Shiki:", 40, 1))