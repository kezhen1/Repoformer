#!/usr/bin/env python
# coding=utf-8

import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from eval_metric import compute_metric_stmt
from eval_metric_cceval import compute_metric_stmt_cceval
from tqdm import tqdm
import time
import torch


def prepare_prompt(tokenizer, task, model_type, left_cxt, right_cxt=None, crossfile_cxt=None):
    prefix_token = '<fim_prefix>'
    suffix_token = '<fim_suffix>'
    middle_token = '<fim_middle>'
    if 'deepseek' in args.model_name_or_path.lower():
        prefix_token = '<｜fim▁begin｜>'
        suffix_token = '<｜fim▁hole｜>'
        middle_token = '<｜fim▁end｜>'
    elif 'qwen' in args.model_name_or_path.lower():
        prefix_token = '<|fim_prefix|>'
        suffix_token = '<|fim_suffix|>'
        middle_token = '<|fim_middle|>'

    if model_type == "codelm_leftright_context":
        # left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - args.right_context_length):])
        # right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        left_cxt_truncated = tokenizer.convert_tokens_to_string(tokenizer.tokenize(left_cxt)[-(args.max_seq_length - args.gen_length - args.right_context_length):])
        right_cxt_truncated = tokenizer.convert_tokens_to_string(tokenizer.tokenize(right_cxt)[:args.right_context_length])
        prompt = f'{prefix_token}{left_cxt_truncated}{suffix_token}{right_cxt_truncated}{middle_token}'
    elif model_type == "codelm":
        # left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length):])
        left_cxt_truncated = tokenizer.convert_tokens_to_string(tokenizer.tokenize(left_cxt)[-(args.max_seq_length - args.gen_length):])
        # prompt = left_cxt_truncated
        prompt = f'{prefix_token}{left_cxt_truncated}{suffix_token}{middle_token}'
    elif model_type == "codelm_cfc":
        assert crossfile_cxt is not None
        # left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - args.cfc_seq_length):])
        # crossfile_cxt_truncated = tokenizer.decode(tokenizer.encode(crossfile_cxt)[:args.cfc_seq_length])
        left_cxt_truncated = tokenizer.convert_tokens_to_string(tokenizer.tokenize(left_cxt)[-(args.max_seq_length - args.gen_length - args.cfc_seq_length):])
        crossfile_cxt_truncated = tokenizer.convert_tokens_to_string(tokenizer.tokenize(crossfile_cxt)[:args.cfc_seq_length])
        # prompt = crossfile_cxt_truncated + '\n\n' + left_cxt_truncated
        prompt = f'{prefix_token}{crossfile_cxt_truncated}\n\n{left_cxt_truncated}{suffix_token}{middle_token}'
    elif model_type == "codelm_right_cfc_left":
        assert crossfile_cxt is not None
        # left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - args.right_context_length - args.cfc_seq_length):])
        # right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        # crossfile_cxt_truncated = tokenizer.decode(tokenizer.encode(crossfile_cxt)[:args.cfc_seq_length])
        left_cxt_truncated = tokenizer.convert_tokens_to_string(tokenizer.tokenize(left_cxt)[-(args.max_seq_length - args.gen_length - args.right_context_length - args.cfc_seq_length):])
        right_cxt_truncated = tokenizer.convert_tokens_to_string(tokenizer.tokenize(right_cxt)[:args.right_context_length])
        crossfile_cxt_truncated = tokenizer.convert_tokens_to_string(tokenizer.tokenize(crossfile_cxt)[:args.cfc_seq_length])
        prompt = f'{prefix_token}{crossfile_cxt_truncated}\n\n{left_cxt_truncated}{suffix_token}{right_cxt_truncated}{middle_token}'
    elif model_type == "codelm_right_cfc_left_v1":
        assert crossfile_cxt is not None
        # left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - args.right_context_length - args.cfc_seq_length):])
        # right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        # crossfile_cxt_truncated = tokenizer.decode(tokenizer.encode('\n\n' + crossfile_cxt)[:args.cfc_seq_length])
        left_cxt_truncated = tokenizer.convert_tokens_to_string(tokenizer.tokenize(left_cxt)[-(args.max_seq_length - args.gen_length - args.right_context_length - args.cfc_seq_length):])
        right_cxt_truncated = tokenizer.convert_tokens_to_string(tokenizer.tokenize(right_cxt)[:args.right_context_length])
        crossfile_cxt_truncated = tokenizer.convert_tokens_to_string(tokenizer.tokenize(crossfile_cxt)[:args.cfc_seq_length])
        prompt = f'{prefix_token}{left_cxt_truncated}{suffix_token}{right_cxt_truncated}{crossfile_cxt_truncated}{middle_token}'
    else:
        raise NotImplementedError

    return prompt


def build_dataset(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    with open(args.prompt_file) as f:
        raw_data = [json.loads(line) for line in f.readlines()]

    data = []
    for entry in raw_data:
        task = args.task
        model_type = args.model_type
        left_cxt = entry["prompt"]
        right_cxt = entry["right_context"]
        crossfile_cxt = None
        if 'crossfile_context' in entry:
            crossfile_cxt = entry["crossfile_context"] if type(entry["crossfile_context"]) == str else entry["crossfile_context"]['text']            
        prompt = prepare_prompt(tokenizer, task, model_type, left_cxt, right_cxt, crossfile_cxt)
        entry['llm_prompt'] = prompt
        data.append(entry)
    
    return data


def process_output(output_ids, tokenizer):
    # output_ids = output.outputs[0].token_ids
    output_text = tokenizer.decode(output_ids)
    res = output_text
    stops = [
        "<｜end▁of▁sentence｜>",
        "<|endoftext|>",
        "<file_sep>",
        "<|file_sep|>",
        "<|fim_pad|>",
        "<|cursor|>",
    ]

    for stop in stops:
        res = res.split(stop)[0]

    return res


def model_inference(args):

    # build data
    data = build_dataset(args)
    prompts = [entry['llm_prompt'] for entry in data]
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # print(prompts[0])
    # exit(0)

    # generate
    # llm = None
    if 'deepseek' in args.model_name_or_path.lower():
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=args.tp_size,
            max_model_len=args.model_max_tokens,
            trust_remote_code=True,
        )
    else:
        # llm = LLM(model=args.model_name_or_path)
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=args.tp_size,
            dtype=args.dtype,
            max_model_len=args.model_max_tokens,
        )

    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=args.gen_length)

    outputs = llm.generate(prompts, sampling_params)
    print(outputs[0])

    all_preds = []
    for entry, output in zip(data, outputs):
    # for entry, output in tqdm(zip(data)):
        start_time = time.time_ns()
        # cur_pred = llm.generate(entry['llm_prompt'], sampling_params, use_tqdm=False)

        # pred = output.outputs[0].text
        pred = process_output(output.outputs[0].token_ids, tokenizer)

        end_time = time.time_ns()
        latency = (end_time - start_time) / 1_000_000_000
        all_preds.append({
            "task_id": entry["metadata"]["task_id"],
            "pred": pred,
            "latency": latency
            # "prompt": entry["llm_prompt"]
        })
        
    with open(f"{args.output_dir}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
        for entry in all_preds:
            f_pred.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--language", type=str, required=True, help="language name")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        default="codelm",
        choices=["codelm", "codelm_cfc", "codelm_leftright_context", 'codelm_right_cfc_left'],
        help="Model type to be loaded"
    )
    parser.add_argument("--prompt_file", type=str, default=None, help="file with a list of prompts")
    parser.add_argument("--gen_length", type=int, default=50, help="max length of generated token sequence")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="max length of prompt")
    parser.add_argument(
        "--cfc_seq_length",
        type=int,
        default=512,
        help="For model_type=codelm_cfc: Text sequence length corresponding to the retrieved nodes"
    )
    parser.add_argument(
        "--right_context_length",
        type=int,
        default=512,
        help="For model_type=codelm_leftright_context: Text sequence length corresponding to the right context"
    )
    parser.add_argument("--output_dir", type=str, default="output_dir", help="output directory to save predictions")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    # compute metric args
    parser.add_argument(
        "--ts_lib",
        type=str,
        default="build/python-lang-parser.so",
        help="tree-sitter lib for tokenize code"
    )
    # only compute metric
    parser.add_argument("--only_compute_metric", action="store_true", help="only compute metric")
    # for cceval metric
    parser.add_argument("--compute_cceval_metric", action='store_true', help="use cceval metric")
    parser.add_argument(
        "--task",
        choices=["line_completion", "api_completion", "function_completion"],
        default="line_completion",
        help="task name"
    )
    parser.add_argument(
        '--tp_size', type=int, default=1,
        help='tensor parallel size'
    )
    parser.add_argument("--dtype", type=str, default='bfloat16', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--model_max_tokens', type=int, default=16384, help='maximum number of tokens of the model')
    args = parser.parse_args()

    print(args)

    model_inference(args)

    if args.compute_cceval_metric:
        compute_metric_stmt_cceval(args)
    else:
        compute_metric_stmt(args)
