import os
import torch
import wandb
import time
import tqdm
import accelerate

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler

from dataset import FeedbackDataset

import argparse

class RM_Trainer:
    def __init__(
        self,
        accelerator: accelerate.Accelerator,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        optimizer: torch.optim.Optimizer,
        weight_dtype: torch.dtype,
        args: argparse.Namespace,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.weight_dtype = weight_dtype
        self.args = args

        if accelerator.is_main_process:
            self.progress_bar = tqdm.tqdm(
                total=self.args.epochs*len(train_dataloader),
                desc="Total Steps",
                leave=False,
            )

            self.run = wandb.init(
                project="convogpt-rm",
                name=f'{self.args.model}-{self.args.epochs}-{self.args.batch_size}-{self.args.learning_rate}--{int(time.time())}',
                config=self.args,
            )

            self.global_step = 0
        
    def save_model(self) -> None:
        self.accelerator.wait_for_everyone()
        path = f'{self.args.output_dir}'
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
            input_ids = batch[0]['input_ids']
            attention_mask = batch[0]['attention_mask']
            reward = batch[1]

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=reward)

            loss = outputs.loss
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        return {
            "train/loss": loss.detach().item(),
            "train/lr": self.lr_scheduler.get_last_lr()[0],
        }
    
    def evaluate(self) -> None:
        self.model.eval()
        for _, batch in enumerate(self.eval_dataloader):
            input_ids = batch[0]['input_ids']
            attention_mask = batch[0]['attention_mask']
            reward = batch[1]

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=reward)

            loss = outputs.loss

            return {
                "eval/loss": loss.detach().item()
            }
    
    def train(self) -> None:
        self.model.train()
        for epoch in range(self.args.epochs):
            for _, batch in enumerate(self.train_dataloader):
                step_start = time.perf_counter()

                metrics = self.step(batch)

                step_end = time.perf_counter()

                if self.accelerator.is_main_process:
                    rank_samples_per_second = self.args.batch_size * (1 / (step_end - step_start))
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
                    
                    if self.global_step % self.args.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        self.run.log(eval_metrics, step=self.global_step)
        self.save_model()

def main():
    parser = argparse.ArgumentParser(description="Supervised GPT finetuning")
    parser.add_argument("--model", type=str, default="hakurei/gpt-j-random-tinier", help="Model name")
    parser.add_argument("--dataset", type=str, default="train.jsonl", help="Training file")
    parser.add_argument("--eval_dataset", type=str, default="valid.jsonl", help="Evaluation file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save model every x steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--learning_rate_schedule", type=str, default="constant", help="Learning rate schedule")
    parser.add_argument("--eval_steps", type=int, default=10, help="Evaluate model every x steps")
    args = parser.parse_args()

    accelerator = accelerate.Accelerator()
    accelerate.utils.set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    def collate_fn(batches):
        input_ids = [
            batch[0]['input_ids'].squeeze(0) for batch in batches
        ]
        attention_mask = tokenizer.pad(
            {"input_ids": input_ids},
            return_tensors="pt",
            padding=True
        )['attention_mask']
        reward = torch.stack([batch[1] for batch in batches])
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": attention_mask,
        }, reward
    
    train_dataset = FeedbackDataset(args.dataset, tokenizer=tokenizer, max_length=tokenizer.model_max_length)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_dataset = FeedbackDataset(args.eval_dataset, tokenizer=tokenizer, max_length=tokenizer.model_max_length)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    model.config.pad_token_id = tokenizer.eos_token_id
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.learning_rate_schedule,
        optimizer=optimizer,
        num_warmup_steps=0.1 * args.epochs * len(train_dataloader),
        num_training_steps=args.epochs * len(train_dataloader),
    )

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    trainer = RM_Trainer(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        lr_scheduler=lr_scheduler,
        optimizer=optimizer,
        weight_dtype=None,
        args=args,
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    print(eval_metrics)

if __name__ == "__main__":
    main()