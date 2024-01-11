from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import os
import argparse
from string import Template
from langchain.llms import Anthropic
from dotenv import dotenv_values
from openai import OpenAI
import torch
import time

ANTHROPIC_API_KEY = dotenv_values(".env").get("ANTHROPIC_API_KEY", None)
OPENAI_API_KEY = dotenv_values(".env").get("OPENAI_API_KEY", None)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip(), strict=False)
            data.append(entry)
    return data

def write_jsonl(data_list, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        for entry in data_list:
            line = json.dumps(entry)
            file.write(line + "\n")

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--device", default="cpu")
parser.add_argument("--api_model", action="store_true")
parser.add_argument("--api_has_logprobs", action="store_true")
parser.add_argument("--cot", action="store_true")
parser.add_argument("--multi_gpu", action="store_true")
parser.add_argument("--half_precision", action="store_true")
parser.add_argument("--impossible_dataset", action="store_true")
args = parser.parse_args()

if args.impossible_dataset:
    evaluation_results_dir = "impossible_evaluation_results"
    data_list = read_jsonl("i_am_an_impossible_dataset.jsonl")
else:
    evaluation_results_dir = "evaluation_results"
    data_list = read_jsonl("i_am_a_strange_dataset.jsonl")

if not args.api_model:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.multi_gpu:
        if args.half_precision:
            model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    else:
        if args.half_precision:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(args.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model.eval()
else:
    if args.model == "claude-2":
        if ANTHROPIC_API_KEY is None:
            raise ValueError("You are trying to use claude-2, but your ANTHROPIC_API_KEY is not set in a .env file.")
        kwargs_for_non_cot = {"max_tokens_to_sample": 2}
        model = Anthropic(model="claude-2", anthropic_api_key=ANTHROPIC_API_KEY)
    elif args.model == "gpt-4" or args.model == "gpt-4-1106-preview" or args.model == "gpt-3.5-turbo":
        if OPENAI_API_KEY is None:
            raise ValueError("You are trying to use gpt-4, but your OPENAI_API_KEY is not set in a .env file.")
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        client = OpenAI()
        kwargs_for_non_cot = {"max_tokens": 2, "top_logprobs": 5, "logprobs": True}
        def model(message, max_tokens=None, logprobs=None, top_logprobs=None, temperature=0):
            completion = client.chat.completions.create(
                temperature=temperature,
                logprobs=logprobs,
                max_tokens=max_tokens,
                top_logprobs=top_logprobs,
                model=args.model,
                messages=[
                    {"role": "user", "content": message}
                ],
            )
            if max_tokens==None:
                return completion.choices[0].message.content
            return completion
    else:
        raise ValueError(f"{args.model} is not a supported model for our evaluation yet when using the api_model option.")


# Compute generation results.
if not args.api_model:
    generation_losses = []
    for datum in tqdm(data_list):
        true_statement = datum["beginning"] + datum["true_continuation"]
        false_statement = datum["beginning"] + datum["false_continuation"]

        true_input_ids = tokenizer(true_statement, return_tensors='pt', truncation=True).input_ids.to(args.device)
        true_loss = model(input_ids=true_input_ids, labels=true_input_ids).loss.item()

        false_input_ids = tokenizer(false_statement, return_tensors='pt', truncation=True).input_ids.to(args.device)
        false_loss = model(input_ids=false_input_ids, labels=false_input_ids).loss.item()

        non_self_referential = datum["non_self_referential_beginning"] is not None and not datum.get("used_in_non_self_referential_prompt", False)

        if non_self_referential:
            non_self_referential_true_statement = datum["non_self_referential_beginning"] + datum["true_continuation"] + "\n\n" + true_statement
            non_self_referential_false_statement = datum["non_self_referential_beginning"] + datum["false_continuation"] + "\n\n" + false_statement

            non_self_referential_true_input_ids = tokenizer(non_self_referential_true_statement, return_tensors='pt', truncation=True).input_ids.to(args.device)
            non_self_referential_true_loss = model(input_ids=non_self_referential_true_input_ids, labels=non_self_referential_true_input_ids).loss.item()

            non_self_referential_false_input_ids = tokenizer(non_self_referential_false_statement, return_tensors='pt', truncation=True).input_ids.to(args.device)
            non_self_referential_false_loss = model(input_ids=non_self_referential_false_input_ids, labels=non_self_referential_false_input_ids).loss.item()
            generation_losses.append({"id": datum["id"], "true": true_loss, "false": false_loss, "non_self_referential_true": non_self_referential_true_loss, "non_self_referential_false": non_self_referential_false_loss})
        else:
            generation_losses.append({"id": datum["id"], "true": true_loss, "false": false_loss})

    write_jsonl(generation_losses, f"{evaluation_results_dir}/{args.model.split('/')[-1]}_generation_losses.jsonl")
    print("Computed generation results")


# Compute validation results
prompt_data_list = read_jsonl("prompt_templates/prompt_data.jsonl")

few_shot_prompt_data_dict = {}
cot_prompt_data_dict = {}
for index in range(len(prompt_data_list)):
    prompt_datum = prompt_data_list[index]
    example_0 = prompt_datum["beginning"] + prompt_datum["true_continuation"]
    example_1 = prompt_datum["beginning"] + prompt_datum["false_continuation"]

    few_shot_prompt_data_dict[f"few_shot_prompt_example_{index}_0"] = example_0
    few_shot_prompt_data_dict[f"few_shot_prompt_example_{index}_1"] = example_1
    few_shot_prompt_data_dict[f"few_shot_prompt_example_{index}_0_answer"] = "True"
    few_shot_prompt_data_dict[f"few_shot_prompt_example_{index}_1_answer"] = "False"

    cot_prompt_data_dict[f"few_shot_prompt_example_{index}_0"] = example_0
    cot_prompt_data_dict[f"few_shot_prompt_example_{index}_1"] = example_1
    cot_prompt_data_dict[f"few_shot_prompt_example_{index}_0_answer"] = prompt_datum["cot_true_answer"]
    cot_prompt_data_dict[f"few_shot_prompt_example_{index}_1_answer"] = prompt_datum["cot_false_answer"]


non_self_referential_prompt_data_list = read_jsonl("prompt_templates/prompt_data_non_self_referential.jsonl")

non_self_referential_few_shot_prompt_data_dict = {}
non_self_referential_cot_prompt_data_dict = {}
for index in range(len(non_self_referential_prompt_data_list)):

    prompt_datum = non_self_referential_prompt_data_list[index]
    example_0 = prompt_datum["beginning"] + prompt_datum["true_continuation"]
    example_1 = prompt_datum["beginning"] + prompt_datum["false_continuation"]

    non_self_referential_example_0 = prompt_datum["non_self_referential_beginning"] + prompt_datum["true_continuation"]
    non_self_referential_example_1 = prompt_datum["non_self_referential_beginning"] + prompt_datum["false_continuation"]

    non_self_referential_few_shot_prompt_data_dict[f"few_shot_prompt_example_{index}_0"] = non_self_referential_example_0 + "\n\n" + example_0
    non_self_referential_few_shot_prompt_data_dict[f"few_shot_prompt_example_{index}_1"] = non_self_referential_example_1 + "\n\n" + example_1
    non_self_referential_few_shot_prompt_data_dict[f"few_shot_prompt_example_{index}_0_answer"] = "True"
    non_self_referential_few_shot_prompt_data_dict[f"few_shot_prompt_example_{index}_1_answer"] = "False"

    non_self_referential_cot_prompt_data_dict[f"few_shot_prompt_example_{index}_0"] = non_self_referential_example_0 + "\n\n" + example_0
    non_self_referential_cot_prompt_data_dict[f"few_shot_prompt_example_{index}_1"] = non_self_referential_example_1 + "\n\n" + example_1
    non_self_referential_cot_prompt_data_dict[f"few_shot_prompt_example_{index}_0_answer"] = prompt_datum["cot_true_answer"]
    non_self_referential_cot_prompt_data_dict[f"few_shot_prompt_example_{index}_1_answer"] = prompt_datum["cot_false_answer"]

cot_template = Template(open("prompt_templates/cot_prompt_template.txt").read())
non_self_referential_cot_template = Template(open("prompt_templates/cot_prompt_template_non_self_referential.txt").read())
few_shot_template = Template(open("prompt_templates/few_shot_prompt_template.txt").read())
non_self_referential_few_shot_template = Template(open("prompt_templates/few_shot_prompt_template_non_self_referential.txt").read())
zero_shot_template = Template(open("prompt_templates/zero_shot_prompt_template.txt").read())
non_self_referential_zero_shot_template = Template(open("prompt_templates/zero_shot_prompt_template_non_self_referential.txt").read())

validation_outputs = []
for datum in tqdm(data_list):

    true_statement = datum["beginning"] + datum["true_continuation"]
    false_statement = datum["beginning"] + datum["false_continuation"]

    true_statement_cot = cot_template.substitute(**cot_prompt_data_dict, example=true_statement)
    false_statement_cot = cot_template.substitute(**cot_prompt_data_dict, example=false_statement)

    non_self_referential = datum["non_self_referential_beginning"] is not None and not datum.get("used_in_non_self_referential_prompt", False)

    if non_self_referential:
        non_self_referential_true_statement = datum["non_self_referential_beginning"] + datum["true_continuation"] + "\n\n" + true_statement
        non_self_referential_false_statement = datum["non_self_referential_beginning"] + datum["false_continuation"] + "\n\n" + false_statement

        non_self_referential_true_statement_cot = non_self_referential_cot_template.substitute(**non_self_referential_cot_prompt_data_dict, example=non_self_referential_true_statement)
        non_self_referential_false_statement_cot = non_self_referential_cot_template.substitute(**non_self_referential_cot_prompt_data_dict, example=non_self_referential_false_statement)

    if args.api_model:

        def get_api_loss(text, generated):
            loss = None
            for obj in generated.choices[0].logprobs.content[0].top_logprobs:
                if obj.token.lower() == text:
                    loss = -obj.logprob
                    break
            return loss

        generated_text_true_statement_cot = model(true_statement_cot, temperature=0)
        generated_text_false_statement_cot = model(false_statement_cot, temperature=0)

        time.sleep(1)  # For rate limits.

        validation_outputs.append({
            "id": datum["id"],
            "generated_text_true_statement_cot": generated_text_true_statement_cot,
            "generated_text_false_statement_cot": generated_text_false_statement_cot,
        })

        if args.api_has_logprobs:
            # Few shot prompt
            true_statement_few_shot = few_shot_template.substitute(**few_shot_prompt_data_dict, example=true_statement, answer="")
            false_statement_few_shot = few_shot_template.substitute(**few_shot_prompt_data_dict, example=false_statement, answer="")

            # Zero shot prompt
            true_statement_zero_shot = zero_shot_template.substitute(example=true_statement, answer="")
            false_statement_zero_shot = zero_shot_template.substitute(example=false_statement, answer="")

            generated_text_true_statement_zero_shot = model(true_statement_zero_shot, temperature=0, **kwargs_for_non_cot)
            generated_text_false_statement_zero_shot = model(false_statement_zero_shot, temperature=0, **kwargs_for_non_cot)

            generated_text_true_statement_few_shot = model(true_statement_few_shot, temperature=0, **kwargs_for_non_cot)
            generated_text_false_statement_few_shot = model(false_statement_few_shot, temperature=0, **kwargs_for_non_cot)

            validation_outputs[-1]["loss_true_statement_true_answer_few_shot"] = get_api_loss("true", generated_text_true_statement_few_shot)
            validation_outputs[-1]["loss_true_statement_false_answer_few_shot"] = get_api_loss("false", generated_text_true_statement_few_shot)
            validation_outputs[-1]["loss_false_statement_true_answer_few_shot"] = get_api_loss("true", generated_text_false_statement_few_shot)
            validation_outputs[-1]["loss_false_statement_false_answer_few_shot"] = get_api_loss("false", generated_text_false_statement_few_shot)
            validation_outputs[-1]["loss_true_statement_true_answer_zero_shot"] = get_api_loss("true", generated_text_true_statement_zero_shot)
            validation_outputs[-1]["loss_true_statement_false_answer_zero_shot"] = get_api_loss("false", generated_text_true_statement_zero_shot)
            validation_outputs[-1]["loss_false_statement_true_answer_zero_shot"] = get_api_loss("true", generated_text_false_statement_zero_shot)
            validation_outputs[-1]["loss_false_statement_false_answer_zero_shot"] = get_api_loss("false", generated_text_false_statement_zero_shot)

        if non_self_referential:

            non_self_referential_generated_text_true_statement_cot = model(non_self_referential_true_statement_cot, temperature=0)
            non_self_referential_generated_text_false_statement_cot = model(non_self_referential_false_statement_cot, temperature=0)

            validation_outputs[-1]["non_self_referential_generated_text_true_statement_cot"] = non_self_referential_generated_text_true_statement_cot
            validation_outputs[-1]["non_self_referential_generated_text_false_statement_cot"] = non_self_referential_generated_text_false_statement_cot

            if args.api_has_logprobs:
                # Few shot prompt
                non_self_referential_true_statement_few_shot = non_self_referential_few_shot_template.substitute(**non_self_referential_few_shot_prompt_data_dict, example=non_self_referential_true_statement, answer="")
                non_self_referential_false_statement_few_shot = non_self_referential_few_shot_template.substitute(**non_self_referential_few_shot_prompt_data_dict, example=non_self_referential_false_statement, answer="")

                # Zero shot prompt
                non_self_referential_true_statement_zero_shot = non_self_referential_zero_shot_template.substitute(example=non_self_referential_true_statement, answer="")
                non_self_referential_false_statement_zero_shot = non_self_referential_zero_shot_template.substitute(example=non_self_referential_false_statement, answer="")

                non_self_referential_generated_text_true_statement_zero_shot = model(non_self_referential_true_statement_zero_shot, temperature=0, **kwargs_for_non_cot)
                non_self_referential_generated_text_false_statement_zero_shot = model(non_self_referential_false_statement_zero_shot, temperature=0, **kwargs_for_non_cot)

                non_self_referential_generated_text_true_statement_few_shot = model(non_self_referential_true_statement_few_shot, temperature=0, **kwargs_for_non_cot)
                non_self_referential_generated_text_false_statement_few_shot = model(non_self_referential_false_statement_few_shot, temperature=0, **kwargs_for_non_cot)

                validation_outputs[-1]["non_self_referential_loss_true_statement_true_answer_few_shot"] = get_api_loss("true", non_self_referential_generated_text_true_statement_few_shot)
                validation_outputs[-1]["non_self_referential_loss_true_statement_false_answer_few_shot"] = get_api_loss("false", non_self_referential_generated_text_true_statement_few_shot)
                validation_outputs[-1]["non_self_referential_loss_false_statement_true_answer_few_shot"] = get_api_loss("true", non_self_referential_generated_text_false_statement_few_shot)
                validation_outputs[-1]["non_self_referential_loss_false_statement_false_answer_few_shot"] = get_api_loss("false", non_self_referential_generated_text_false_statement_few_shot)
                validation_outputs[-1]["non_self_referential_loss_true_statement_true_answer_zero_shot"] = get_api_loss("true", non_self_referential_generated_text_true_statement_zero_shot)
                validation_outputs[-1]["non_self_referential_loss_true_statement_false_answer_zero_shot"] = get_api_loss("false", non_self_referential_generated_text_true_statement_zero_shot)
                validation_outputs[-1]["non_self_referential_loss_false_statement_true_answer_zero_shot"] = get_api_loss("true", non_self_referential_generated_text_false_statement_zero_shot)
                validation_outputs[-1]["non_self_referential_loss_false_statement_false_answer_zero_shot"] = get_api_loss("false", non_self_referential_generated_text_false_statement_zero_shot)
            
        # Write the outputs separately here as we go, because APIs are expensive and if they go down, we still want something to show for it.
        write_jsonl(validation_outputs, f"{args.model.split('/')[-1]}_validation_outputs.jsonl")
    else:
        # Few shot prompt
        true_statement_true_answer_few_shot = few_shot_template.substitute(**few_shot_prompt_data_dict, example=true_statement, answer="True")
        true_statement_false_answer_few_shot = few_shot_template.substitute(**few_shot_prompt_data_dict, example=true_statement, answer="False")

        false_statement_true_answer_few_shot = few_shot_template.substitute(**few_shot_prompt_data_dict, example=false_statement, answer="True")
        false_statement_false_answer_few_shot = few_shot_template.substitute(**few_shot_prompt_data_dict, example=false_statement, answer="False")

        # Zero shot prompt
        true_statement_true_answer_zero_shot = zero_shot_template.substitute(example=true_statement, answer="True")
        true_statement_false_answer_zero_shot = zero_shot_template.substitute(example=true_statement, answer="False")

        false_statement_true_answer_zero_shot = zero_shot_template.substitute(example=false_statement, answer="True")
        false_statement_false_answer_zero_shot = zero_shot_template.substitute(example=false_statement, answer="False")

        def run_zs_or_fs(true_statement_true_answer, true_statement_false_answer, false_statement_true_answer, false_statement_false_answer):
            input_ids_true_statement_true_answer = tokenizer(true_statement_true_answer, return_tensors="pt", truncation=True).input_ids.to(args.device)
            input_ids_true_statement_false_answer = tokenizer(true_statement_false_answer, return_tensors="pt", truncation=True).input_ids.to(args.device)
            input_ids_false_statement_true_answer = tokenizer(false_statement_true_answer, return_tensors="pt", truncation=True).input_ids.to(args.device)
            input_ids_false_statement_false_answer = tokenizer(false_statement_false_answer,  return_tensors="pt", truncation=True).input_ids.to(args.device)

            loss_true_statement_true_answer = model(input_ids=input_ids_true_statement_true_answer, labels=input_ids_true_statement_true_answer).loss.item()
            loss_true_statement_false_answer = model(input_ids=input_ids_true_statement_false_answer, labels=input_ids_true_statement_false_answer).loss.item()
            loss_false_statement_true_answer = model(input_ids=input_ids_false_statement_true_answer, labels=input_ids_false_statement_true_answer).loss.item()
            loss_false_statement_false_answer = model(input_ids=input_ids_false_statement_false_answer, labels=input_ids_false_statement_false_answer).loss.item()
            
            return loss_true_statement_true_answer, loss_true_statement_false_answer, loss_false_statement_true_answer, loss_false_statement_false_answer

        loss_true_statement_true_answer_zero_shot, loss_true_statement_false_answer_zero_shot, loss_false_statement_true_answer_zero_shot, loss_false_statement_false_answer_zero_shot = run_zs_or_fs(true_statement_true_answer_zero_shot, true_statement_false_answer_zero_shot, false_statement_true_answer_zero_shot, false_statement_false_answer_zero_shot)
        loss_true_statement_true_answer_few_shot, loss_true_statement_false_answer_few_shot, loss_false_statement_true_answer_few_shot, loss_false_statement_false_answer_few_shot = run_zs_or_fs(true_statement_true_answer_few_shot, true_statement_false_answer_few_shot, false_statement_true_answer_few_shot, false_statement_false_answer_few_shot)
        
        validation_outputs.append({
            "id": datum["id"],
            "loss_true_statement_true_answer_few_shot": loss_true_statement_true_answer_few_shot,
            "loss_true_statement_false_answer_few_shot": loss_true_statement_false_answer_few_shot,
            "loss_false_statement_true_answer_few_shot": loss_false_statement_true_answer_few_shot,
            "loss_false_statement_false_answer_few_shot": loss_false_statement_false_answer_few_shot,
            "loss_true_statement_true_answer_zero_shot": loss_true_statement_true_answer_zero_shot,
            "loss_true_statement_false_answer_zero_shot": loss_true_statement_false_answer_zero_shot,
            "loss_false_statement_true_answer_zero_shot": loss_false_statement_true_answer_zero_shot,
            "loss_false_statement_false_answer_zero_shot": loss_false_statement_false_answer_zero_shot,
        })

        if non_self_referential:
            # Few shot prompt
            non_self_referential_true_statement_true_answer_few_shot = non_self_referential_few_shot_template.substitute(**non_self_referential_few_shot_prompt_data_dict, example=non_self_referential_true_statement, answer="True")
            non_self_referential_true_statement_false_answer_few_shot = non_self_referential_few_shot_template.substitute(**non_self_referential_few_shot_prompt_data_dict, example=non_self_referential_true_statement, answer="False")

            non_self_referential_false_statement_true_answer_few_shot = non_self_referential_few_shot_template.substitute(**non_self_referential_few_shot_prompt_data_dict, example=non_self_referential_false_statement, answer="True")
            non_self_referential_false_statement_false_answer_few_shot = non_self_referential_few_shot_template.substitute(**non_self_referential_few_shot_prompt_data_dict, example=non_self_referential_false_statement, answer="False")

            # Zero shot prompt
            non_self_referential_true_statement_true_answer_zero_shot = non_self_referential_zero_shot_template.substitute(example=non_self_referential_true_statement, answer="True")
            non_self_referential_true_statement_false_answer_zero_shot = non_self_referential_zero_shot_template.substitute(example=non_self_referential_true_statement, answer="False")

            non_self_referential_false_statement_true_answer_zero_shot = non_self_referential_zero_shot_template.substitute(example=non_self_referential_false_statement, answer="True")
            non_self_referential_false_statement_false_answer_zero_shot = non_self_referential_zero_shot_template.substitute(example=non_self_referential_false_statement, answer="False")

            non_self_referential_loss_true_statement_true_answer_zero_shot, non_self_referential_loss_true_statement_false_answer_zero_shot, non_self_referential_loss_false_statement_true_answer_zero_shot, non_self_referential_loss_false_statement_false_answer_zero_shot = run_zs_or_fs(non_self_referential_true_statement_true_answer_zero_shot, non_self_referential_true_statement_false_answer_zero_shot, non_self_referential_false_statement_true_answer_zero_shot, non_self_referential_false_statement_false_answer_zero_shot)
            non_self_referential_loss_true_statement_true_answer_few_shot, non_self_referential_loss_true_statement_false_answer_few_shot, non_self_referential_loss_false_statement_true_answer_few_shot, non_self_referential_loss_false_statement_false_answer_few_shot = run_zs_or_fs(non_self_referential_true_statement_true_answer_few_shot, non_self_referential_true_statement_false_answer_few_shot, non_self_referential_false_statement_true_answer_few_shot, non_self_referential_false_statement_false_answer_few_shot)
        
            validation_outputs[-1]["non_self_referential_loss_true_statement_true_answer_zero_shot"] = non_self_referential_loss_true_statement_true_answer_zero_shot
            validation_outputs[-1]["non_self_referential_loss_true_statement_false_answer_zero_shot"] = non_self_referential_loss_true_statement_false_answer_zero_shot
            validation_outputs[-1]["non_self_referential_loss_false_statement_true_answer_zero_shot"] = non_self_referential_loss_false_statement_true_answer_zero_shot
            validation_outputs[-1]["non_self_referential_loss_false_statement_false_answer_zero_shot"] = non_self_referential_loss_false_statement_false_answer_zero_shot

            validation_outputs[-1]["non_self_referential_loss_true_statement_true_answer_few_shot"] = non_self_referential_loss_true_statement_true_answer_few_shot
            validation_outputs[-1]["non_self_referential_loss_true_statement_false_answer_few_shot"] = non_self_referential_loss_true_statement_false_answer_few_shot
            validation_outputs[-1]["non_self_referential_loss_false_statement_true_answer_few_shot"] = non_self_referential_loss_false_statement_true_answer_few_shot
            validation_outputs[-1]["non_self_referential_loss_false_statement_false_answer_few_shot"] = non_self_referential_loss_false_statement_false_answer_few_shot

        # COT prompt
        if args.cot:
            def run_cot(true_statement_cot, false_statement_cot):
                input_ids_true_statement_cot = tokenizer(true_statement_cot, return_tensors="pt", truncation=True).input_ids.to(args.device)
                input_ids_false_statement_cot = tokenizer(false_statement_cot, return_tensors="pt", truncation=True).input_ids.to(args.device)
                max_new_tokens = min(200, max(0, model.config.max_position_embeddings - input_ids_true_statement_cot.shape[-1]), max(0, model.config.max_position_embeddings - input_ids_false_statement_cot.shape[-1]))  # Maximum length of the generated tokens
                if max_new_tokens == 0:
                    return "", ""
                prompt_length_true_statement_cot = len(input_ids_true_statement_cot[0])
                generated_ids_true_statement_cot = model.generate(input_ids_true_statement_cot, max_new_tokens=max_new_tokens, num_return_sequences=1, num_beams=1, do_sample=False, pad_token_id=tokenizer.eos_token_id, temperature=0.0)
                generated_text_true_statement_cot = tokenizer.decode(generated_ids_true_statement_cot[0][prompt_length_true_statement_cot:], skip_special_tokens=True)

                prompt_length_false_statement_cot = len(input_ids_false_statement_cot[0])
                generated_ids_false_statement_cot = model.generate(input_ids_false_statement_cot, max_new_tokens=max_new_tokens, num_return_sequences=1, num_beams=1, do_sample=False, pad_token_id=tokenizer.eos_token_id, temperature=0.0)
                generated_text_false_statement_cot = tokenizer.decode(generated_ids_false_statement_cot[0][prompt_length_false_statement_cot:], skip_special_tokens=True)
                return generated_text_true_statement_cot, generated_text_false_statement_cot

            generated_text_true_statement_cot, generated_text_false_statement_cot = run_cot(true_statement_cot, false_statement_cot)
            validation_outputs[-1]["generated_text_true_statement_cot"] = generated_text_true_statement_cot
            validation_outputs[-1]["generated_text_false_statement_cot"] = generated_text_false_statement_cot

            if non_self_referential:
                non_self_referential_generated_text_true_statement_cot, non_self_referential_generated_text_false_statement_cot = run_cot(non_self_referential_true_statement_cot, non_self_referential_false_statement_cot)
                validation_outputs[-1]["non_self_referential_generated_text_true_statement_cot"] = non_self_referential_generated_text_true_statement_cot
                validation_outputs[-1]["non_self_referential_generated_text_false_statement_cot"] = non_self_referential_generated_text_false_statement_cot

write_jsonl(validation_outputs, f"{evaluation_results_dir}/{args.model.split('/')[-1]}_validation_outputs.jsonl")
print("Computed validation results")


