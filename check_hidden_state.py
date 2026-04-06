import os
import torch
from load_mlp import load_mlp_by_weights
from utils.llm import Generater
from utils.data import QADataset
from utils.utils import write_jsonl, read_json
import argparse


def get_args(params: dict):

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=params["source"] or 'data/nq/nq-dev-test.jsonl')
    parser.add_argument('--response', type=str, default='')
    parser.add_argument('--usechat', action='store_true')
    parser.add_argument('--type', type=str, choices=['qa', 'qa_rag', 'qa_evidence', 'qa_cot', 'qa_more', 'qa_extract', 'qa_prior'], default=params["type"] or 'qa')
    parser.add_argument('--ra', type=str, default=params["ra"] or "none", choices=ra_dict.keys())
    parser.add_argument('--outfile', type=str, default='data/qa/chatgpt-nq-none.json')
    parser.add_argument('--idx', type=str, default="")
    parser.add_argument('--model_path', type=str, default=params["model_path"] or "")
    parser.add_argument('--batch_size', type=int, default=params["batch_size"] or 1)
    parser.add_argument('--task', type=str, default=params["task"] or 'nq')
    parser.add_argument('--max_new_tokens', type=int, default=params["max_new_tokens"] or 64)
    parser.add_argument('--hidden_states', type=bool, default=params["hidden_states"] or False)
    parser.add_argument('--output_states', type=bool, default=False)
    parser.add_argument('--attn_weights', type=bool, default=False)
    parser.add_argument('--hidden_idx_mode', type=str, default=params["hidden_idx_mode"] or 'last')
    parser.add_argument('--need_layers', type=str, default=params["need_layers"] or 'last', choices=['all', 'last', 'mid'])
    parser.add_argument('--gpu_device', type=str, default=params["gpu_device"] or '7')
    parser.add_argument('--weight_path', type=str, default=params["weight_path"])
    parser.add_argument('--hidden_prob_output_dir', type=str, default=params["hidden_prob_output_dir"])
    args = parser.parse_args()
    args.model_name = args.model_path.split('/')[-1].replace('_', '-').lower()

    return args


def get_hidden_state(args):
    all_data = QADataset(args)
    engine = Generater(args)
    engine.load_data(all_data)
    res, _, __ = engine.get_res()
    
    # Save full response data as JSONL
    output_dir = args.hidden_prob_output_dir
    os.makedirs(output_dir, exist_ok=True)
    write_jsonl(res, os.path.join(output_dir, "hidden_state_by_llm.jsonl"))
    print(f"Saved full response data to {os.path.join(output_dir, 'hidden_state_by_llm.jsonl')}")
    
    # Save hidden states for reuse
    hidden_states_list = []
    for _res in res:
        hidden_state = _res['hidden_states']['first']
        hidden_states_list.append(hidden_state)
    hidden_states_tensor = torch.tensor(hidden_states_list)
    torch.save(hidden_states_tensor, os.path.join(output_dir, "dev_hidden_states.pt"))
    print(f"Saved hidden states to {os.path.join(output_dir, 'dev_hidden_states.pt')}")
    
    return res


def get_hidden_prob_by_mlp(res, gpu_device, weight_path, output_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mlp_by_weights(gpu_device, weight_path)
    result = []
    for _res in res:
        input_data = torch.tensor(_res["hidden_states"]["first"], dtype=torch.float32)
        input_data = input_data.to(device)
        output = model(input_data)
        print(output)
        result.append(output.tolist())
    write_jsonl(result, os.path.join(output_dir, "hidden_prob_by_mlp.jsonl"))
    write_jsonl(res, os.path.join(output_dir, "hidden_state_by_llm.jsonl"))



def run(params: dict):
    args = get_args(params or get_default_params())
    hidden_states = get_hidden_state(args)
    get_hidden_prob_by_mlp(hidden_states, args.gpu_device, args.weight_path, args.hidden_prob_output_dir)


def generate_rag_prompt(question: str, context: str) -> str:
    prompt = f"""
                **System Role:**
                You are a rigorous language model. Please answer the question based on the provided context. 
                If the context does not support reasoning about the answer, please answer the question based on your own knowledge.
                **Context:**
                {context}
                **Question:** 
                {question}
            """.strip()
    return prompt


def convert_jsonl_by_rag(meta_path, target_path):
    meta_data = read_json(meta_path)
    target_data = []
    for item in meta_data:
        item["question"] = generate_rag_prompt(item["question"], item["context"])
        target_data.append(item)
    write_jsonl(target_data, target_path)
