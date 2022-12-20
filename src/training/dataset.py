import torch
import argparse
import numpy as np
import transformers
import json

def decode(in_file: str, out_file: str, tokenizer: transformers.AutoTokenizer) -> int:
    mem = np.memmap(in_file, mode="r", dtype="uint16")
    tokens = len(mem)
    with open(out_file, "a") as f:
        for token in mem:
            f.write(tokenizer.decode([token]))
    return tokens

def encode(in_file: str, out_file: str, tokenizer: transformers.AutoTokenizer) -> int:
    with open(in_file, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    with open(out_file, "wb") as f:
        for token in tokens:
            f.write(np.uint16(token))
    return len(tokens)

class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, in_file: str, chunks: int, tokenizer: transformers.AutoTokenizer):
        self.in_file = in_file
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.mem = np.memmap(in_file, mode="r", dtype="uint16")

    def __len__(self):
        return len(self.mem) // self.chunks
    
    def __getitem__(self, idx):
        return torch.tensor(self.mem[idx*self.chunks:(idx+1)*self.chunks])

class FeedbackDataset(torch.utils.data.Dataset):
    def __init__(self, feedback_file: str, tokenizer: transformers.AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feedback_file = feedback_file
        
        with open(feedback_file) as f:
            self.feedback = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.feedback)

    def __getitem__(self, idx):
        feedback = self.feedback[idx]
        feedback_input = '\n'.join(feedback["input"].split("\n")[-2:])
        feedback_str = f'{feedback_input} {feedback["output"].lstrip().rstrip()}'
        seq = self.tokenizer(
            feedback_str,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
            )
        reward = torch.tensor([feedback["reward"]]).unsqueeze(0)
        return seq, reward

# sft file example
# {
#     "input": "Anonymous: Hi, how are you?\nGPT:",
#     "output": " I'm good, how are you?\n",
#     "reward": 0.0
# }
class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, sft_file: str, tokenizer: transformers.AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sft_file = sft_file
        
        with open(sft_file) as f:
            self.sft = [json.loads(line) for line in f]
        
        # iterate over sft, removing any that have a reward of 0
        self.sft = [sft for sft in self.sft if sft["reward"] != 0.0]

    def __len__(self):
        return len(self.sft)
    
    def __getitem__(self, idx):
        sft = self.sft[idx]
        sft_input_tokens = self.tokenizer(sft["input"], return_tensors="pt").input_ids
        sft_output_tokens = self.tokenizer(f' {sft["output"].lstrip().rstrip()}', return_tensors="pt").input_ids
        input_ids = torch.cat([sft_input_tokens, sft_output_tokens], dim=-1)
        start_positions = torch.tensor([len(sft_input_tokens[0])])
        end_positions = torch.tensor([len(sft_input_tokens[0]) + len(sft_output_tokens[0]) - 1])
        return {
            "input_ids": input_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Creator')
    parser.add_argument('--in_file', type=str, help='input file to use', required=True)
    parser.add_argument('--out_file', type=str, help='output file to use', required=True)
    parser.add_argument('--model', type=str, help='model tokenizer to use', required=True)
    args = parser.parse_args()

    encode(args.in_file, args.out_file, transformers.AutoTokenizer.from_pretrained(args.model))