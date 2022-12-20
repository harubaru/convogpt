import torch

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
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

    # create a loss mask where we ignore the loss for the tokens that are not in the answer
    loss_mask = torch.zeros_like(logits[:, :, 0])
    for i in range(len(start_positions)):
        loss_mask[i, start_positions[i]:end_positions[i]+1] = 1
    
    answer_logits = logits[:, start_positions[0]:end_positions[0]+1]
    answer_input_ids = input_ids[:, start_positions[0]:end_positions[0]+1]

    # compute loss for prompt and answer
    loss = torch.nn.functional.cross_entropy(answer_logits.view(-1, answer_logits.size(-1)), answer_input_ids.view(-1), ignore_index=-1)

    if not return_dict:
        output = (loss,) + outputs[2:]
        return ((loss,) + outputs[2:]) if return_dict else output

    return CausalLMOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

if __name__ == '__main__':
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
