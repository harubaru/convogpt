import os
import torch
import accelerate
import tqdm
import time
import argparse
import wandb

from dataset import TokenizedDataset

from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

class UFT_Trainer:
    def __init__(
        self,
        accelerator: accelerate.Accelerator,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        train_dataloader: torch.utils.data.DataLoader,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        optimizer: torch.optim.Optimizer,
        weight_dtype: torch.dtype,
        args: argparse.Namespace,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
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
                project="convogpt-uftlm",
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

            try:
                outputs = self.model(**batch)

                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                print(f"input_ids: {input_ids}")
                print(f"attention_mask: {attention_mask}")
                print('Skipping batch...')
                loss = torch.tensor(float('nan'), device=self.accelerator.device)
        
        return {
            "train/loss": loss.detach().item(),
            "train/lr": self.lr_scheduler.get_last_lr()[0],
        }
    
    def train(self) -> None:
        self.model.train()
        for epoch in range(self.args.epochs):
            for _, batch in enumerate(self.train_dataloader):
                step_start = time.perf_counter()

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

    parser = argparse.ArgumentParser(description="Unsupervised GPT finetuning")
    parser.add_argument("--model", type=str, default="hakurei/gpt-j-random-tinier", help="Model name")
    parser.add_argument("--dataset", type=str, default="train.jsonl", help="Training file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save model every x steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--learning_rate_schedule", type=str, default="constant", help="Learning rate schedule")
    args = parser.parse_args()

    accelerator = accelerate.Accelerator()
    accelerate.utils.set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    collator = lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                            'attention_mask': torch.stack([f[1] for f in data]),
                            'labels': torch.stack([f[0] for f in data])}
    
    train_dataset = TokenizedDataset(args.dataset, context_length=tokenizer.model_max_length)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    optim_cls = torch.optim.AdamW
    try:
        import bitsandbytes as bnb
        optim_cls = bnb.optim.AdamW8bit
    except ImportError:
        pass

    optimizer = optim_cls(model.parameters(), lr=args.learning_rate)

    lr_scheduler = get_scheduler(
        name=args.learning_rate_schedule,
        optimizer=optimizer,
        num_warmup_steps=0.1 * args.epochs * len(train_dataloader),
        num_training_steps=args.epochs * len(train_dataloader),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    trainer = UFT_Trainer(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        lr_scheduler=lr_scheduler,
        optimizer=optimizer,
        weight_dtype=None,
        args=args,
    )

    trainer.train()

if __name__ == "__main__":
    main()
