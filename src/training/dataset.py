import os
import struct
import random
import torch
import argparse
import numpy as np
import transformers
import json
import tqdm
from typing import Tuple

def decode(in_file: str, out_file: str, tokenizer: transformers.AutoTokenizer) -> int:
    mem = np.memmap(in_file, mode="r", dtype="uint16")
    tokens = len(mem)
    with open(out_file, "a") as f:
        for token in tqdm.tqdm(mem):
            f.write(tokenizer.decode([token]))
    return tokens

def encode(in_file: str, out_file: str, tokenizer: transformers.AutoTokenizer) -> None:
    with open(out_file, "ab") as w:
        with open(in_file, "r", encoding="utf-8") as f:
            for line in tqdm.tqdm(f):
                w.write(np.uint16(tokenizer.encode(line)))

class TokenizedDataset(torch.utils.data.Dataset):
    """
    Consumes a flat binary file containing 16-bit token serialization, aligned
    along `context_length` chunks.
    """

    def __init__(self, path: str, context_length: int = 2048):
        file_stat = os.stat(path)
        self.file = open(path, 'rb')
        self.length = int(file_stat.st_size / 2 / context_length)
        self.formatstr = '%sH' % context_length
        self.context_length = context_length
        length_mb = os.stat(path).st_size / 1024.0 / 1024.0
        num_tokens = self.length * context_length
        print(f"DATASET: {path}")
        print(f"DATASET SIZE: {length_mb:,.2f}mb, {num_tokens:,} tokens, "
              f"{self.length:,} contexts")

    def __len__(self) -> int:
        return self.length

    def load(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.seek(idx)
        input_ids = torch.tensor(
            struct.unpack(self.formatstr,
                          self.file.read(self.context_length * 2)))
        mask = torch.zeros(self.context_length)
        return input_ids, mask

    def seek(self, idx):
        self.file.seek(self.context_length * idx * 2)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.load(idx)

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
import tqdm
class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, sft_file: str, tokenizer: transformers.AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sft_file = sft_file
        
        with open(sft_file) as f:
            self.sft = [json.loads(line) for line in f]
        
        # iterate over sft, removing any that have a reward of 0
        i = 0
        for sft in self.sft:
            i += 1
            print(i, sft)
            if sft['reward'] != 1.0:
                self.sft.remove(sft)
        #self.sft = [sft for sft in self.sft if sft["reward"] != 0.0]

        # iterate over sft, removing any that have too many tokens
        for feedback in tqdm.tqdm(self.sft, desc="Validating SFT"):
            inputs = feedback["input"] + f' {feedback["output"].lstrip().rstrip()}\n'
            if len(self.tokenizer(inputs).input_ids) > self.max_length:
                self.sft.remove(feedback)
                print(f"Removed {feedback['output']} due to length")

    def __len__(self):
        return len(self.sft)
    
    def __getitem__(self, idx):
        sft = self.sft[idx]
        sft_input_tokens = self.tokenizer(sft["input"], return_tensors="pt").input_ids
        sft_output_tokens = self.tokenizer(f' {sft["output"].lstrip().rstrip()}\n', return_tensors="pt").input_ids
        input_ids = torch.cat([sft_input_tokens, sft_output_tokens], dim=-1)
        start_positions = torch.tensor([len(sft_input_tokens[0])])
        end_positions = torch.tensor([len(sft_input_tokens[0]) + len(sft_output_tokens[0]) - 1])
        return {
            "input_ids": input_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }


class ChatMLDataset(torch.utils.data.Dataset):
    def __init__(self, chat_file: str, tokenizer, max_length: int, message_start_token: str, message_end_token: str,
                 ignore_index: int, optimize_assistant: bool):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.message_start_token = message_start_token
        self.message_end_token = message_end_token
        self.conversations = []
        self.IGNORE_INDEX = ignore_index
        self.optimize_assistant = optimize_assistant

        try:
            with open(f"{chat_file}.cache", "rb") as f:
                print("Loading cached dataset")
                self.conversations = torch.load(f)
        except FileNotFoundError:
            self._load_and_preprocess_data(chat_file)
            torch.save(self.conversations, f=open(f"{chat_file}.cache", "wb"))
            print("Cached dataset")

        random.shuffle(self.conversations)

    def debug_print_conversation(self, idx):
        if idx >= len(self.conversations):
            print("Index out of range.")
            return

        conversation = self.conversations[idx]
        input_ids = conversation['inputs']
        labels = conversation['targets']

        decoded_inputs = [self.tokenizer.decode(token_id) for token_id in input_ids]
        decoded_labels = [self.tokenizer.decode(token_id) if token_id != self.IGNORE_INDEX else 'IGNORE_INDEX' for
                          token_id in labels]

        print(f"Conversation {idx}:")
        for token_id, token, label in zip(input_ids, decoded_inputs, decoded_labels):
            print(f"Token ({token_id}): {token} - Label: {label}")

    def _load_and_preprocess_data(self, file_path):
        with open(file_path, 'r') as file:
            for line in tqdm.tqdm(file):
                conversation = json.loads(line)
                self._preprocess_conversation(conversation["conversation"])

    def _preprocess_conversation(self, messages):
        inputs = []
        targets = []

        # Find all the indices for assistant messages
        assistant_indices = [i for i, msg in enumerate(messages) if msg['role'] == 'assistant']

        # If there are no assistant messages, we can't form a valid conversation for our use case
        if not assistant_indices:
            return

        for i, message in enumerate(messages):
            start_role = f"{self.message_start_token}{message['role']}\n"
            content = f"{message['content']}{self.message_end_token}"

            if i == 0:
                start_tokens = self.tokenizer.encode(start_role, add_special_tokens=True)
            else:
                start_tokens = self.tokenizer.encode(start_role, add_special_tokens=False)
            content_tokens = self.tokenizer.encode(content, add_special_tokens=False)

            if i == len(messages) - 1:
                content_tokens += [self.tokenizer.eos_token_id]

            tokenized_content = start_tokens + content_tokens

            label = [self.IGNORE_INDEX] * len(tokenized_content)

            if self.optimize_assistant:
                if message['role'] == 'assistant':
                    label[-len(content_tokens):] = content_tokens

            if i == len(messages) - 1:
                label[-len(content_tokens):] = content_tokens
            else:
                newline_tokens = self.tokenizer.encode('\n', add_special_tokens=False)
                tokenized_content += newline_tokens
                label += [self.IGNORE_INDEX] * len(newline_tokens)

            inputs.extend(tokenized_content)
            targets.extend(label)

        self.conversations.append({'inputs': inputs, 'targets': targets})

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        input_ids = conversation['inputs']
        labels = conversation['targets']

        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels
        }

class InstructDataset(torch.utils.data.Dataset):
    def __init__(self, json_file: str, tokenizer, max_length: int, ignore_index: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruct_data = []
        self.IGNORE_INDEX = ignore_index
        self._load_data(json_file)

    def _load_data(self, file_path):
        # Load the data from the json file
        data = None
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for instruction in data:
            # Tokenize the instruction and input text
            instruction_inputs = instruction["input"]
            if instruction_inputs != "":
                instruction_inputs += "\n\n"
            
            instruction_tokens = self.tokenizer.encode(instruction["instruction"] + "\n\n" + instruction_inputs, add_special_tokens=True, truncation=True, max_length=self.max_length)
            
            output_tokens = self.tokenizer.encode(instruction["output"], add_special_tokens=True, truncation=True, max_length=self.max_length)
            
            print(instruction["instruction"] + "\n\n" + instruction_inputs + instruction["output"])

            tokenized_instruction = instruction_tokens + output_tokens

            label = [self.IGNORE_INDEX] * len(tokenized_instruction)
            label[-len(output_tokens):] = output_tokens

            self.instruct_data.append({
                'input_ids': tokenized_instruction,
                'labels': label,
            })

    def debug_print_conversation(self, idx):
        if idx >= len(self.instruct_data):
            print("Index out of range.")
            return

        instruction = self.instruct_data[idx]
        input_ids = instruction['input_ids']
        labels = instruction['labels']

        decoded_inputs = [self.tokenizer.decode(token_id) for token_id in input_ids]
        decoded_labels = [self.tokenizer.decode(token_id) if token_id != self.IGNORE_INDEX else 'IGNORE_INDEX' for token_id in labels]

        print(f"Instruction {idx}:")
        for token, label in zip(decoded_inputs, decoded_labels):
            print(f"Token: {token} - Label: {label}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.instruct_data[idx]

        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(input_ids.ne(self.tokenizer.pad_token_id), dtype=torch.long)
        labels = torch.tensor(item['labels'], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

if __name__ == '__main__':
    from transformers import AutoTokenizer
    dataset = ChatMLDataset(
        'dataset/convo-dataset-v0.jsonl',
        AutoTokenizer.from_pretrained('models/convo-7B-002/checkpoint-200', use_fast=False),
        4096,
        '<|im_start|>',
        '<|im_end|>',
        -100,
        True
    )
    dataset.debug_print_conversation(1)
