import os
import torch
import accelerate
import tqdm
import time
import argparse
import wandb

from dataset import TokenizedDataset, FeedbackDataset, SFTDataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput

from typing import Union, Optional

# Supervised Finetuning: Compute loss between model output and target using start_positions and end_positions
def sft_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    start_positions: Optional[torch.LongTensor] = None,
    end_positions: Optional[torch.LongTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[torch.Tensor, CausalLMOutput]:
    try:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    except AttributeError:
        return_dict = True

    outputs = self.transformer(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = outputs[0]

    logits = self.lm_head(sequence_output)

    answer_logits = logits[:, start_positions[0]:end_positions[0]+1]
    answer_input_ids = input_ids[:, start_positions[0]:end_positions[0]+1]

    # compute loss for prompt and answer
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
    shift_answer_logits = answer_logits[..., :-1, :].contiguous()
    shift_answer_labels = answer_input_ids[..., 1:].contiguous()
    answer_loss = loss_fct(shift_answer_logits.view(-1, answer_logits.size(-1)), shift_answer_labels.view(-1))

    loss = answer_loss

    if not return_dict:
        output = (loss,) + outputs[2:]
        return ((loss,) + outputs[2:]) if return_dict else output

    return CausalLMOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

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
                total=self.args.epochs*len(train_dataloader),
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
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            path = f'{self.args.output_dir}/{self.run.name}'
            os.makedirs(path, exist_ok=True)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(path, save_function=self.accelerator.save)

    def step(self, batch: dict) -> None:
        with self.accelerator.accumulate(self.model):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']

            try:
                outputs = sft_forward(
                    self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )

                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                print(f"input_ids: {input_ids}")
                print(f"attention_mask: {attention_mask}")
                print(f"start_positions: {start_positions}")
                print(f"end_positions: {end_positions}")
                print('Skipping batch...')
                loss = torch.tensor(float('nan'), device=self.accelerator.device)
        
        return {
            "train/loss": loss.detach().item(),
        }
    
    def train(self) -> None:
        self.model.train()
        for epoch in range(self.args.epochs):
            for _, batch in enumerate(self.train_dataloader):
                step_start = time.perf_counter()

                #print(f"####\n{self.tokenizer.decode(batch['input_ids'][0])}\n#{batch['start_positions'][0]}:{batch['end_positions'][0]}\n####")

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
    parser.add_argument("--model", type=str, default="hakurei/gpt-j-random-tinier", help="Model name")
    parser.add_argument("--dataset", type=str, default="train.jsonl", help="Training file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save model every x steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    accelerator = accelerate.Accelerator()
    accelerate.utils.set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    def collate_fn(batches):
        input_ids = [
            batch["input_ids"].squeeze(0) for batch in batches
        ]
        padded_tokens = tokenizer.pad(
            {"input_ids": input_ids}, return_tensors="pt", padding=True
        )
        start_positions = torch.stack(
            [batch["start_positions"] for batch in batches]
        )
        end_positions = torch.stack(
            [batch["end_positions"] for batch in batches]
        )
        return {
            "input_ids": padded_tokens["input_ids"],
            "attention_mask": padded_tokens["attention_mask"],
            "start_positions": start_positions,
            "end_positions": end_positions,
        }
    
    train_dataset = SFTDataset(args.dataset, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

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
