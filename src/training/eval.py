import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

data = open("spanish_lit_dataset.txt", "r").read()
tokenizer = AutoTokenizer.from_pretrained("hakurei/bloom-1b1-arb-thesis")
encodings = tokenizer(data, return_tensors="pt", padding=True)
model = AutoModelForCausalLM.from_pretrained("hakurei/bloom-1b1-arb-thesis").to("cuda")

# evaluate
nlls = []
model.eval()
prev_end_loc = 0
seq_len = encodings.input_ids.size(1)
stride = 512
with torch.no_grad():
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + 2048, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood.item())

        prev_end_loc = end_loc
        if end_loc == seq_len:
            print(f"begin_loc: {begin_loc}, end_loc: {end_loc}, seq_len: {seq_len}")
            break

ppl_loss = torch.tensor(nlls).sum() / seq_len
ppl = torch.exp(ppl_loss)
print(f"Perplexity: {ppl.item()}\nPerplexity loss: {ppl_loss.item()}")
