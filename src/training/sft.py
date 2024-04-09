import os
import torch
import accelerate
import tqdm
import time
import argparse
import wandb

from dataset import TokenizedDataset, FeedbackDataset, SFTDataset, ChatMLDataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput

from typing import Union, Optional


class SFT_Trainer:
    def __init__(
            self,
            accelerator: accelerate.Accelerator,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            train_dataloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            weight_dtype: torch.dtype,
            args: argparse.Namespace,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.weight_dtype = weight_dtype
        self.args = args

        if accelerator.is_main_process:
            self.progress_bar = tqdm.tqdm(
                total=self.args.epochs * len(train_dataloader),
                desc="Total Steps",
                leave=False,
            )

            self.run = wandb.init(
                project="convogpt-sftlm",
                name=f'{self.args.model}-{self.args.epochs}-{self.args.batch_size}-{self.args.learning_rate}--{int(time.time())}',
                config=self.args,
            )

            self.global_step = 0

    def save_model(self) -> None:
        path = f'{self.args.output_dir}/{self.run.name}'
        os.makedirs(path, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            path,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save
        )
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(path)

    def step(self, batch: dict) -> None:
        with self.accelerator.accumulate(self.model):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            try:
                outputs = self.model.forward(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                print('Skipping batch...')
                return {"train/loss": torch.nan}

        return {
            "train/loss": loss.detach().item(),
        }

    def train(self) -> None:
        self.model.train()
        for epoch in range(self.args.epochs):
            for _, batch in enumerate(self.train_dataloader):
                step_start = time.perf_counter()

                # print(f"####\n{self.tokenizer.decode(batch['input_ids'][0])}\n#{batch['start_positions'][0]}:{batch['end_positions'][0]}\n####")

                metrics = self.step(batch)

                step_end = time.perf_counter()

                if self.accelerator.is_main_process:
                    rank_samples_per_second = self.args.batch_size / (step_end - step_start)
                    world_samples_per_second = rank_samples_per_second * self.accelerator.num_processes

                    metrics.update({
                        "perf/rank_samples_per_second": rank_samples_per_second,
                        "perf/world_samples_per_second": world_samples_per_second,
                        "train/epoch": epoch,
                        "train/step": self.global_step,
                        "train/samples_seen": self.global_step * self.args.batch_size,
                    })

                    self.global_step += 1

                    self.progress_bar.update(1)
                    self.progress_bar.set_postfix(**metrics)

                    self.run.log(metrics, step=self.global_step)

                    if self.global_step % self.args.save_steps == 0:
                        self.save_model()
        self.accelerator.wait_for_everyone()
        self.save_model()


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised GPT finetuning")
    parser.add_argument("--model", type=str, default="models/convobase-125m", help="Model name")
    parser.add_argument("--dataset", type=str, default="dataset/ultrachat-1k.jsonl", help="Training file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save model every x steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient Accumulation Steps")
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation
    )
    accelerate.utils.set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    def collate_fn(batches):
        input_ids = [batch["input_ids"].squeeze(0) for batch in batches]
        labels = [batch["labels"].squeeze(0) for batch in batches]

        # Padding for 'input_ids' and creating 'attention_mask'
        padded_tokens = tokenizer.pad(
            {"input_ids": input_ids}, return_tensors="pt", padding=True
        )

        # Padding for 'labels' - we can use the same padding index as for the 'input_ids'
        padded_labels = tokenizer.pad(
            {"input_ids": labels}, return_tensors="pt", padding=True
        )["input_ids"]  # 'pad_to_multiple_of' may be required for some models like OPT

        return {
            "input_ids": padded_tokens["input_ids"],
            "attention_mask": padded_tokens["attention_mask"],
            "labels": padded_labels
        }

    train_dataset = ChatMLDataset(args.dataset, tokenizer, 2048, "<|im_start|>", "<|im_end|>", -100)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model)
    optim_cls = torch.optim.AdamW
    try:
        import bitsandbytes as bnb
        optim_cls = bnb.optim.AdamW8bit
    except ImportError:
        pass

    optimizer = optim_cls(model.parameters(), lr=args.learning_rate)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    trainer = SFT_Trainer(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        weight_dtype=None,
        args=args,
    )

    trainer.train()


if __name__ == '__main__':
    """
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained('distilgpt2')
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

    # Add supervised finetuning forward method to model
    model.forward = sft_forward.__get__(model)

    # Create input tensors
    question = 'What is the capital of France?'
    answer = 'The capital of France is Paris.'
    question_tokens = tokenizer.encode(question, return_tensors='pt')
    answer_tokens = tokenizer.encode(answer, return_tensors='pt')
    input_ids = torch.cat([question_tokens, answer_tokens], dim=-1)

    start_positions = torch.tensor([len(question_tokens[0])])
    end_positions = torch.tensor([len(question_tokens[0]) + len(answer_tokens[0]) - 1])

    # Compute loss
    loss = model(input_ids, start_positions=start_positions, end_positions=end_positions).loss
    print(loss)
    """
    main()
