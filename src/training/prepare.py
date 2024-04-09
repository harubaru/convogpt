import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description='Model Preparer')
parser.add_argument('--model', type=str, help='input model to use', required=True)
parser.add_argument('--output', type=str, help='output model', required=True)
args = parser.parse_args()

CONVO_START="<|im_start|>"
CONVO_END="<|im_end|>"

def embedding_resize(
    tokens: list,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
):
    num_new_tokens = tokenizer.add_special_tokens(
        {'pad_token': '<pad>', 'additional_special_tokens': [CONVO_START, CONVO_END]}
    )
    model.resize_token_embeddings(len(tokenizer))

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

    input_embeddings[-num_new_tokens:] = input_embeddings_avg
    output_embeddings[-num_new_tokens:] = output_embeddings_avg

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model)

embedding_resize([CONVO_START, CONVO_END], tokenizer, model)
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

tokenizer.save_pretrained(args.output)
model.save_pretrained(args.output)
